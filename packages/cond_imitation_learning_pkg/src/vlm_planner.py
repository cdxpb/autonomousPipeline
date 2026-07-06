"""
Duckiebot high-level planner — SmolVLM2 + FSM only.

Pipeline: raw image → SmolVLM2 (yes/no oracle) → FSM → intent

Usage:
    python vlm_planner.py --instruction "take the second left, first right and stop" \
                          --images data/images/ \
                          --precision fp16          # fp16 | int8 | int4

Quantisation options:
    fp16  — half precision, default, works on CUDA + MPS (M1 Mac)

"""

import re, argparse, time
from pathlib import Path
from typing  import List, Tuple, Dict, Optional
from enum    import Enum, auto
from dataclasses import dataclass, field

from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    model_id:  str = "HuggingFaceTB/SmolVLM2-500M-Instruct"
    # 256M is faster; 2B is more accurate — swap freely
    # "HuggingFaceTB/SmolVLM2-256M-Instruct"
    # "HuggingFaceTB/SmolVLM2-2B-Instruct"

    precision: str = "fp16"   # fp16 | int8 | int4

    # FSM windows
    at_intersection_window:   int = 3   # consecutive YES frames to confirm at intersection
    left_intersection_window: int = 4   # consecutive NO  frames to confirm left intersection
    cooldown_frames:          int = 15  # suppress after executing a turn

    # Intent labels (must match your dataset)
    intents: List[str] = field(default_factory=lambda:
        ["lane_following", "straight", "left", "right", "stop"])


# ─────────────────────────────────────────────────────────────────────────────
# Instruction parser
# ─────────────────────────────────────────────────────────────────────────────

ORDINALS = {
    "first":1, "1st":1, "one":1,
    "second":2,"2nd":2, "two":2,
    "third":3, "3rd":3, "three":3,
    "fourth":4,"4th":4, "four":4,
    "next":1,
}
DIRECTIONS = {
    "left":"left", "right":"right",
    "straight":"straight", "forward":"straight", "ahead":"straight",
}
STOP_WORDS = {"stop", "halt", "end", "park", "finish"}


def parse_instruction(text: str) -> List[Tuple[str, int]]:
    """
    'take the second left, first right and stop'
    → [('left', 2), ('right', 1), ('stop', 0)]
    """
    plan   = []
    chunks = re.split(r"[,]|\band\b|\bthen\b", text.lower())
    for chunk in chunks:
        words = chunk.split()
        if not words:
            continue
        if any(w in STOP_WORDS for w in words):
            plan.append(("stop", 0))
            continue
        direction = next((DIRECTIONS[w] for w in words if w in DIRECTIONS), None)
        if direction is None:
            continue
        count = next((ORDINALS[w] for w in words if w in ORDINALS), 1)
        plan.append(("straight", count) if direction == "straight" else (direction, count))
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Navigation FSM
# ─────────────────────────────────────────────────────────────────────────────

class FSMState(Enum):
    LANE_FOLLOWING          = auto()   # emit 'lane_following'
    PASSING_THROUGH_VALID   = auto()   # at target intersection but not target count; emit 'straight'
    PASSING_THROUGH_INVALID = auto()   # at intersection but target direction invalid; emit 'straight'
    EXECUTING               = auto()   # at target intersection; emit planned direction
    DONE                    = auto()   # emit 'stop'


class NavFSM:
    """
    Consumes one boolean per frame (at_intersection?) and emits intent.

    The VLM answers "is the robot at an intersection right now?"
    The FSM handles the plan logic: which intersection to act on, what to do.
    """

    def __init__(self, plan: List[Tuple[str, int]], cfg: Config):
        self.plan     = list(plan)
        self.cfg      = cfg
        self.state    = FSMState.LANE_FOLLOWING
        self.count    = 0     # valid intersections confirmed so far
        self.valid_buf = []   # sliding window for 'yes'
        self.invalid_buf = [] # sliding window for 'pass'
        self.none_buf = []    # sliding window for 'no'
        self.cooldown = 0
        self.frame    = 0
        self.log: List[dict] = []

        if not self.plan:
            self.state = FSMState.DONE

    @property
    def _current(self) -> Tuple[str, int]:
        return self.plan[0] if self.plan else ("stop", 0)

    def _confirmed(self, buf: list, window: int) -> bool:
        return len(buf) >= window and sum(buf[-window:]) == window

    def step(self, ans: str) -> str:
        self.frame += 1

        if self.state == FSMState.DONE:
            return self._emit("stop", ans)

        if self.cooldown > 0:
            self.cooldown -= 1
            self.valid_buf, self.invalid_buf, self.none_buf = [], [], []
            return self._emit("lane_following", ans)

        # ── LANE_FOLLOWING ────────────────────────────────────────────────────
        if self.state == FSMState.LANE_FOLLOWING:
            if ans == "yes":
                self.valid_buf.append(1)
                self.invalid_buf = []
            elif ans == "pass":
                self.invalid_buf.append(1)
                self.valid_buf = []
            else:
                self.valid_buf = []
                self.invalid_buf = []

            if self._confirmed(self.valid_buf, self.cfg.at_intersection_window):
                self.count += 1
                self.valid_buf = []
                action, target = self._current
                if action == "stop":
                    self.state = FSMState.DONE
                elif self.count >= target:
                    self.state = FSMState.EXECUTING
                    self.count = 0
                else:
                    self.state = FSMState.PASSING_THROUGH_VALID
            elif self._confirmed(self.invalid_buf, self.cfg.at_intersection_window):
                self.invalid_buf = []
                self.state = FSMState.PASSING_THROUGH_INVALID
                
            return self._emit("lane_following", ans)

        # ── PASSING_THROUGH_VALID / INVALID ───────────────────────────────────
        if self.state in (FSMState.PASSING_THROUGH_VALID, FSMState.PASSING_THROUGH_INVALID):
            if ans == "no":
                self.none_buf.append(1)
            else:
                self.none_buf = []
                
            if self._confirmed(self.none_buf, self.cfg.left_intersection_window):
                self.state    = FSMState.LANE_FOLLOWING
                self.none_buf = []
            return self._emit("straight", ans)

        # ── EXECUTING ─────────────────────────────────────────────────────────
        if self.state == FSMState.EXECUTING:
            action, _ = self._current
            if ans == "no":
                self.none_buf.append(1)
            else:
                self.none_buf = []
                
            if self._confirmed(self.none_buf, self.cfg.left_intersection_window):
                self.plan.pop(0)
                nxt, _ = self._current
                self.state    = FSMState.DONE if (nxt == "stop" or not self.plan) \
                                else FSMState.LANE_FOLLOWING
                self.cooldown = self.cfg.cooldown_frames
                self.none_buf = []
            return self._emit(action, ans)

        return self._emit("lane_following", ans)

    def _emit(self, intent: str, ans: str) -> str:
        self.log.append(dict(
            frame=self.frame, state=self.state.name,
            ans=ans, intent=intent,
            count=self.count, remaining=list(self.plan),
        ))
        return intent

    @property
    def done(self) -> bool:
        return self.state == FSMState.DONE


# ─────────────────────────────────────────────────────────────────────────────
# SmolVLM2 oracle
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "This is a forward-facing camera image from a Duckietown autonomous robot driving on a road. "
    "Look at the road ahead. "
    "If the robot is NOT at an intersection, answer 'no'. "
    "If the robot is at an intersection and it is possible to go {direction}, answer 'yes'. "
    "If the robot is at an intersection but it is IMPOSSIBLE to go {direction}, answer 'pass'. "
    "Answer with exactly one word: yes, no, or pass."
)


def _get_device():
    import torch
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


class VLMOracle:
    """
    SmolVLM2 constrained to binary yes/no: is the robot at an intersection?

    Quantisation
    ────────────
    fp16  works on CUDA and MPS (M1 Mac).
    int8  requires CUDA + bitsandbytes.
    int4  requires CUDA + bitsandbytes; use BitsAndBytesConfig(load_in_4bit=True).

    Logit masking
    ─────────────
    Instead of generating text, we compare the logit of the first yes-token
    vs the first no-token. This is deterministic, fast (no sampling), and
    immune to hallucination at the output level.
    """

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        
        self.device = _get_device()
        dtype       = torch.float16 if self.device.type != "cpu" else torch.float32
        print(f"Loading {cfg.model_id}  [{cfg.precision}]  on {self.device} ...")

        bnb_cfg = None
        if cfg.precision == "int4":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif cfg.precision == "int8":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            quantization_config=bnb_cfg,
            device_map="auto" if bnb_cfg else None,
        )
        if bnb_cfg is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        # Resolve yes/no/pass token IDs once at load time
        self.yes_ids  = self._token_ids(["yes", "Yes", "YES"])
        self.no_ids   = self._token_ids(["no",  "No",  "NO"])
        self.pass_ids = self._token_ids(["pass", "Pass", "PASS"])
        print(f"Ready. yes={self.yes_ids} no={self.no_ids} pass={self.pass_ids}")

    def _token_ids(self, words: List[str]) -> List[int]:
        ids = set()
        for w in words:
            ids.update(self.processor.tokenizer.encode(w, add_special_tokens=False))
        return list(ids)

    def at_intersection(self, image: Image.Image, direction: str = "straight") -> Tuple[str, float]:
        """
        Returns (answer: str, confidence: float).
        Answer is one of 'yes', 'no', 'pass'.
        Confidence is the softmax probability of the chosen class.
        """
        import torch
        
        prompt_text = PROMPT_TEMPLATE.format(direction=direction)
        
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        
        with torch.no_grad():
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            logits   = self.model(**inputs).logits[0, -1, :]   # next-token logits
            yes_logit  = logits[self.yes_ids].max().item()
            no_logit   = logits[self.no_ids ].max().item()
            pass_logit = logits[self.pass_ids].max().item()
            
            probs = torch.softmax(torch.tensor([yes_logit, no_logit, pass_logit]), dim=0)
            max_idx = probs.argmax().item()
            conf = probs[max_idx].item()
            ans = ["yes", "no", "pass"][max_idx]

        return ans, conf


# ─────────────────────────────────────────────────────────────────────────────
# Full navigator
# ─────────────────────────────────────────────────────────────────────────────

class Navigator:
    def __init__(self, instruction: str, cfg: Config):
        plan      = parse_instruction(instruction)
        self.fsm  = NavFSM(plan, cfg)
        self.vlm  = VLMOracle(cfg)
        print(f"\nInstruction : '{instruction}'")
        print(f"Plan        : {plan}\n")

    def step(self, image: Image.Image) -> dict:
        current_target = self.fsm._current[0] if not self.fsm.done else "straight"
        if current_target == "stop":
            current_target = "straight"
            
        ans, conf = self.vlm.at_intersection(image, direction=current_target)
        intent    = self.fsm.step(ans)
        
        return dict(
            intent          = intent,
            at_intersection = ans,
            confidence      = conf,
            fsm_state       = self.fsm.state.name,
            done            = self.fsm.done,
        )

    def run(self, image_paths: List[Path]) -> List[dict]:
        results = []
        for path in image_paths:
            t0     = time.perf_counter()
            image  = Image.open(path).convert("RGB")
            result = self.step(image)
            result["ms"]    = round((time.perf_counter() - t0) * 1000, 1)
            result["frame"] = path.name
            results.append(result)
            print(f"  {path.name:30s}  "
                  f"ans={result['at_intersection']:4s} "
                  f"({result['confidence']:.2f})  "
                  f"→ {result['intent']:14s}  "
                  f"[{result['fsm_state']}]  "
                  f"{result['ms']} ms")
            if result["done"]:
                print("  FSM done — instruction complete.")
                break
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Duckiebot VLM planner")
    parser.add_argument("--instruction", required=True,
                        help='e.g. "take the second left, first right and stop"')
    parser.add_argument("--images",  required=True,
                        help="Directory of images or a single image path")
    parser.add_argument("--precision", default="fp16",
                        choices=["fp16", "int8", "int4"])
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--at-window",   type=int, default=3)
    parser.add_argument("--left-window", type=int, default=4)
    parser.add_argument("--cooldown",    type=int, default=15)
    args = parser.parse_args()

    cfg = Config(
        model_id                  = args.model,
        precision                 = args.precision,
        at_intersection_window    = args.at_window,
        left_intersection_window  = args.left_window,
        cooldown_frames           = args.cooldown,
    )

    # Collect images
    p = Path(args.images)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png"}
        paths = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    else:
        paths = [p]

    print(f"Images: {len(paths)}")
    nav     = Navigator(args.instruction, cfg)
    results = nav.run(paths)

    # Summary
    from collections import Counter
    counts = Counter(r["intent"] for r in results)
    avg_ms = sum(r["ms"] for r in results) / len(results) if results else 0
    print(f"\nSummary — {len(results)} frames processed")
    print(f"  Intent breakdown: {dict(counts)}")
    print(f"  Avg latency: {avg_ms:.0f} ms/frame")


if __name__ == "__main__":
    main()

