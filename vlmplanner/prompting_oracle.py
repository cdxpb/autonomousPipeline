# zero-shot and few-shot SmolVLM2 prompting, no training. same APPROACH_PROMPT/EXIT_PROMPT
# vlm_planner.py already uses live (these are the "proper" prompts, tuned against real
# footage, see that file's docstring) -- this just adds a few-shot variant and evaluates
# both against the real annotated val set. see NOTES.local.md.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                        / "packages" / "cond_imitation_learning_pkg" / "src"))
from vlm_planner import Config, VLMOracle, APPROACH_PROMPT, EXIT_PROMPT  # noqa: E402


class ZeroShotOracle(VLMOracle):
    """VLMOracle, but do_image_splitting=False. VLMOracle's own _classify() uses the
    processor's default (splitting on), which multiplies each image into several tiles
    and made a single call take 2-3s; a few-shot call with several images the same way
    took 30-170s, intractable for a real eval set. Doesn't touch vlm_planner.py itself,
    this is an eval-time speed choice, not a change to the live prompting path."""

    def _classify(self, image, prompt_text, class_tokens):
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        with torch.no_grad():
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], do_image_splitting=False, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits[0, -1, :]
            labels = list(class_tokens.keys())
            class_logits = [logits[ids].max().item() for ids in class_tokens.values()]
            probs = torch.softmax(torch.tensor(class_logits), dim=0)
            max_idx = probs.argmax().item()
        return labels[max_idx], probs[max_idx].item()


class FewShotOracle(VLMOracle):
    """Same prompts as VLMOracle, with a few labeled exemplars prepended as prior
    turns before the real question, one call each for approach_exemplars (list of
    (PIL.Image, direction, "yes"|"pass"|"no")) and exit_exemplars ((PIL.Image, "yes"|"no"))."""

    def __init__(self, cfg: Config, approach_exemplars, exit_exemplars):
        super().__init__(cfg)
        self.approach_exemplars = approach_exemplars
        self.exit_exemplars = exit_exemplars

    def _classify_fewshot(self, image, prompt_text, exemplars, class_tokens):
        import torch

        messages, images = [], []
        for ex_img, ex_prompt, ex_answer in exemplars:
            messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex_prompt}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ex_answer}]})
            images.append(ex_img)
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]})
        images.append(image)

        with torch.no_grad():
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images, do_image_splitting=False, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits[0, -1, :]
            labels = list(class_tokens.keys())
            class_logits = [logits[ids].max().item() for ids in class_tokens.values()]
            probs = torch.softmax(torch.tensor(class_logits), dim=0)
            max_idx = probs.argmax().item()
        return labels[max_idx], probs[max_idx].item()

    def classify_approach(self, image, direction: str = "straight"):
        prompt_text = APPROACH_PROMPT.format(direction=direction)
        exemplars = [(img, APPROACH_PROMPT.format(direction=d), ans) for img, d, ans in self.approach_exemplars]
        return self._classify_fewshot(image, prompt_text, exemplars,
                                       {"yes": self.yes_ids, "pass": self.pass_ids, "no": self.no_ids})

    def classify_exit(self, image):
        exemplars = [(img, EXIT_PROMPT, ans) for img, ans in self.exit_exemplars]
        return self._classify_fewshot(image, EXIT_PROMPT, exemplars,
                                       {"yes": self.yes_ids, "no": self.no_ids})
