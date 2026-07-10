"""
Duckiebot high-level planner: SmolVLM2 + FSM.

Pipeline: raw image -> SmolVLM2 (two-question oracle) -> FSM -> intent

Usage:
    python vlm_planner.py --instruction "take the second left, first right and stop" \
                          --images data/images/ --precision fp16   # fp16 | int8 | int4
"""

import re, argparse, time
from pathlib import Path
from typing  import List, Tuple, Dict, Optional
from enum    import Enum, auto
from dataclasses import dataclass, field

from PIL import Image


# --- Config ---

@dataclass
class Config:
    model_id:  str = "HuggingFaceTB/SmolVLM2-500M-Instruct"
    # 256M is faster, 2B is more accurate, swap freely:
    # "HuggingFaceTB/SmolVLM2-256M-Instruct"
    # "HuggingFaceTB/SmolVLM2-2B-Instruct"

    precision: str = "fp16"   # fp16 | int8 | int4

    # asymmetric on purpose: quick to confirm a junction, quicker to confirm exit.
    # tune against real footage via notebooks/vlm_prompt_tuning.ipynb
    at_intersection_window: int = 2   # consecutive confirming frames to trigger approach/at
    exit_window:             int = 2   # consecutive confirming frames to trigger exit
    cooldown_frames:         int = 15  # suppress re-triggering right after executing a turn

    # Intent labels (must match your dataset)
    intents: List[str] = field(default_factory=lambda:
        ["lane_following", "straight", "left", "right", "stop"])


# --- Instruction parser ---

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
    -> [('left', 2), ('right', 1), ('stop', 0)]
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


# --- Navigation FSM ---

class FSMState(Enum):
    LANE_FOLLOWING          = auto()   # emit 'lane_following'; asks the APPROACH question
    PASSING_THROUGH_VALID   = auto()   # at target junction but not target count; emit 'straight'
    PASSING_THROUGH_INVALID = auto()   # at junction but target direction invalid; emit 'straight'
    EXECUTING               = auto()   # at target junction; emit planned direction
    DONE                    = auto()   # emit 'stop'


class NavFSM:
    """
    Consumes one VLM answer per frame and emits an intent.

    `needs()` tells the caller which question to ask before calling step(): LANE_FOLLOWING
    expects an APPROACH answer ('yes'/'no'/'pass'); every other non-terminal state expects
    an EXIT answer ('yes'=clear, 'no'=still at the junction).
    """

    def __init__(self, plan: List[Tuple[str, int]], cfg: Config):
        self.plan     = list(plan)
        self.cfg      = cfg
        self.state    = FSMState.LANE_FOLLOWING
        self.count    = 0     # valid junctions confirmed so far
        self.yes_buf     = []   # sliding window for approach 'yes'
        self.invalid_buf = []   # sliding window for approach 'pass'
        self.exit_buf    = []   # sliding window for exit 'yes' (clear)
        self.cooldown = 0
        self.frame    = 0
        self.log: List[dict] = []

        if not self.plan:
            self.state = FSMState.DONE

    @property
    def _current(self) -> Tuple[str, int]:
        return self.plan[0] if self.plan else ("stop", 0)

    @property
    def done(self) -> bool:
        return self.state == FSMState.DONE

    def needs(self) -> Optional[str]:
        """Which VLM question ('approach' | 'exit') the caller should ask this frame,
        or None if no VLM call is needed at all (done or in cooldown)."""
        if self.done or self.cooldown > 0:
            return None
        return "approach" if self.state == FSMState.LANE_FOLLOWING else "exit"

    def _confirmed(self, buf: list, window: int) -> bool:
        return len(buf) >= window and sum(buf[-window:]) == window

    def step(self, ans: Optional[str]) -> str:
        self.frame += 1

        if self.state == FSMState.DONE:
            return self._emit("stop", ans)

        if self.cooldown > 0:
            self.cooldown -= 1
            self.yes_buf, self.invalid_buf, self.exit_buf = [], [], []
            return self._emit("lane_following", ans)

        if self.state == FSMState.LANE_FOLLOWING:
            return self._step_approach(ans)

        if self.state in (FSMState.PASSING_THROUGH_VALID, FSMState.PASSING_THROUGH_INVALID):
            return self._step_passing_through(ans)

        if self.state == FSMState.EXECUTING:
            return self._step_executing(ans)

        return self._emit("lane_following", ans)

    def _step_approach(self, ans: str) -> str:
        if ans == "yes":
            self.yes_buf.append(1)
            self.invalid_buf = []
        elif ans == "pass":
            self.invalid_buf.append(1)
            self.yes_buf = []
        else:
            self.yes_buf = []
            self.invalid_buf = []

        if self._confirmed(self.yes_buf, self.cfg.at_intersection_window):
            self.count += 1
            self.yes_buf = []
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

    def _step_passing_through(self, ans: str) -> str:
        if ans == "yes":
            self.exit_buf.append(1)
        else:
            self.exit_buf = []

        if self._confirmed(self.exit_buf, self.cfg.exit_window):
            self.state    = FSMState.LANE_FOLLOWING
            self.exit_buf = []
        return self._emit("straight", ans)

    def _step_executing(self, ans: str) -> str:
        action, _ = self._current
        if ans == "yes":
            self.exit_buf.append(1)
        else:
            self.exit_buf = []

        if self._confirmed(self.exit_buf, self.cfg.exit_window):
            self.plan.pop(0)
            nxt, _ = self._current
            self.state    = FSMState.DONE if (nxt == "stop" or not self.plan) \
                            else FSMState.LANE_FOLLOWING
            self.cooldown = self.cfg.cooldown_frames
            self.exit_buf = []
        return self._emit(action, ans)

    def _emit(self, intent: str, ans: Optional[str]) -> str:
        self.log.append(dict(
            frame=self.frame, state=self.state.name,
            ans=ans, intent=intent,
            count=self.count, remaining=list(self.plan),
        ))
        return intent


# --- SmolVLM2 oracle ---

APPROACH_PROMPT = (
    "This is a forward-facing camera image from a small robot driving along a road with "
    "painted lane markings: a yellow dashed line down the center and solid white lines "
    "along the outer edges. Where two or more roads meet, a 4-way crossing or a "
    "T-junction, the lane markings open up (the lines stop, fork, or are crossed by a "
    "red or orange stop line) and the far road(s) become visible. "
    "Look at the road ahead of the robot, including anything visible in the distance, "
    "not just directly under the robot. "
    "If a junction like this is visible ahead OR the robot is currently at one, AND "
    "driving {direction} from there leads onto a real, open road (not blocked by grass, "
    "a wall, or empty space), answer 'yes'. "
    "If a junction is visible ahead OR the robot is currently at one, but {direction} is "
    "NOT an open road from there (for example a T-junction where that side has no road), "
    "answer 'pass'. "
    "If there is no junction visible ahead or nearby and the robot is simply following "
    "its lane (including a gentle curve), answer 'no'. "
    "Answer with exactly one word: yes, pass, or no."
)

EXIT_PROMPT = (
    "This is a forward-facing camera image from a small robot driving along a road with "
    "painted lane markings: a yellow dashed center line and solid white outer edge "
    "lines. The robot recently entered a junction (a 4-way crossing or T-junction, where "
    "the lane markings open up, a stop line is often painted across the road, and other "
    "roads become visible). "
    "Look at the road immediately at and around the robot right now. "
    "If the robot is back on a normal single lane, a continuous yellow dashed center "
    "line with white edge lines on both sides, with no open junction area or stop line "
    "at the robot's current position, answer 'yes'. "
    "If the robot is still inside, crossing, or right at the junction (open road area, a "
    "stop line, or multiple lanes visibly converging at the robot), answer 'no'. "
    "Answer with exactly one word: yes or no."
)


def _get_device():
    import torch
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


class VLMOracle:
    """
    SmolVLM2 constrained to fixed-vocabulary answers via logit comparison (no free
    generation). Two public methods, one per FSM question: `classify_approach`,
    `classify_exit`.

    precision: fp16 works on CUDA and MPS. int8/int4 require CUDA + bitsandbytes.
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

        # Resolve yes/no/pass token IDs once at load time. Both prompts share this
        # same fixed vocabulary (exit just never uses 'pass') so there's only one set
        # of token IDs to resolve.
        self.yes_ids  = self._token_ids(["yes", "Yes", "YES"])
        self.no_ids   = self._token_ids(["no",  "No",  "NO"])
        self.pass_ids = self._token_ids(["pass", "Pass", "PASS"])
        print(f"Ready. yes={self.yes_ids} no={self.no_ids} pass={self.pass_ids}")

    def _token_ids(self, words: List[str]) -> List[int]:
        ids = set()
        for w in words:
            ids.update(self.processor.tokenizer.encode(w, add_special_tokens=False))
        return list(ids)

    def _classify(self, image: Image.Image, prompt_text: str,
                  class_tokens: Dict[str, List[int]]) -> Tuple[str, float]:
        """One forward pass -> (label, confidence), argmax'd over class_tokens."""
        import torch

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]

        with torch.no_grad():
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            logits = self.model(**inputs).logits[0, -1, :]   # next-token logits
            labels = list(class_tokens.keys())
            class_logits = [logits[ids].max().item() for ids in class_tokens.values()]

            probs   = torch.softmax(torch.tensor(class_logits), dim=0)
            max_idx = probs.argmax().item()

        return labels[max_idx], probs[max_idx].item()

    def classify_approach(self, image: Image.Image, direction: str = "straight") -> Tuple[str, float]:
        """Is a junction ahead-or-here, and is `direction` open there? -> yes / pass / no."""
        prompt_text = APPROACH_PROMPT.format(direction=direction)
        return self._classify(image, prompt_text,
                               {"yes": self.yes_ids, "pass": self.pass_ids, "no": self.no_ids})

    def classify_exit(self, image: Image.Image) -> Tuple[str, float]:
        """Has the robot cleared the junction and returned to lane-following? -> yes / no."""
        return self._classify(image, EXIT_PROMPT,
                               {"yes": self.yes_ids, "no": self.no_ids})


# --- Full navigator ---

class Navigator:
    def __init__(self, instruction: str, cfg: Config):
        plan      = parse_instruction(instruction)
        self.fsm  = NavFSM(plan, cfg)
        self.vlm  = VLMOracle(cfg)
        print(f"\nInstruction : '{instruction}'")
        print(f"Plan        : {plan}\n")

    def step(self, image: Image.Image) -> dict:
        query = self.fsm.needs()

        if query == "approach":
            direction = self.fsm._current[0]
            if direction == "stop":
                direction = "straight"
            ans, conf = self.vlm.classify_approach(image, direction=direction)
        elif query == "exit":
            ans, conf = self.vlm.classify_exit(image)
        else:
            ans, conf = None, 0.0

        intent = self.fsm.step(ans)

        return dict(
            intent     = intent,
            vlm_query  = query,
            vlm_answer = ans,
            confidence = conf,
            fsm_state  = self.fsm.state.name,
            done       = self.fsm.done,
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
            query_str  = result["vlm_query"] or "-"
            answer_str = result["vlm_answer"] or "-"
            print(f"  {path.name:30s}  "
                  f"{query_str:8s} ans={answer_str:4s} "
                  f"({result['confidence']:.2f})  "
                  f"-> {result['intent']:14s}  "
                  f"[{result['fsm_state']}]  "
                  f"{result['ms']} ms")
            if result["done"]:
                print("  FSM done, instruction complete.")
                break
        return results


# --- CLI ---

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
    parser.add_argument("--at-window",   type=int, default=2,
                        help="consecutive confirming frames to trigger approach/at")
    parser.add_argument("--exit-window", type=int, default=2,
                        help="consecutive confirming frames to trigger exit")
    parser.add_argument("--cooldown",    type=int, default=15)
    args = parser.parse_args()

    cfg = Config(
        model_id                = args.model,
        precision                = args.precision,
        at_intersection_window   = args.at_window,
        exit_window               = args.exit_window,
        cooldown_frames           = args.cooldown,
    )

    p = Path(args.images)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png"}
        paths = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    else:
        paths = [p]

    print(f"Images: {len(paths)}")
    nav     = Navigator(args.instruction, cfg)
    results = nav.run(paths)

    from collections import Counter
    counts = Counter(r["intent"] for r in results)
    avg_ms = sum(r["ms"] for r in results) / len(results) if results else 0
    print(f"\nSummary: {len(results)} frames processed")
    print(f"  Intent breakdown: {dict(counts)}")
    print(f"  Avg latency: {avg_ms:.0f} ms/frame")


if __name__ == "__main__":
    main()
