from .tc_adv_zprompt_encode import advanced_zprompt_encode

class TC_ADV_ZPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),

                # scheduled
                "use_schedule": ("BOOLEAN", {"default": True}),
                "schedule_steps": ("INT", {"default": 30, "min": 1, "max": 10000}),

                # Z-Image tokenizer/model options
                "max_sequence_length": ("INT", {"default": 1024, "min": 16, "max": 32768}),
                "enable_thinking": ("BOOLEAN", {"default": False}),

                # weighting options
                "weight_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.01}),
                "clamp_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.01}),

                # AND mixing (embedding-level)
                "and_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_bias": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/advanced"

    def encode(
        self,
        clip,
        text,
        use_schedule,
        schedule_steps,
        max_sequence_length,
        enable_thinking,
        weight_strength,
        clamp_min,
        clamp_max,
        and_strength,
        base_bias,
    ):
        cond = advanced_zprompt_encode(
            clip=clip,
            text=text,
            use_schedule=use_schedule,
            schedule_steps=schedule_steps,
            max_sequence_length=max_sequence_length,
            enable_thinking=enable_thinking,
            weight_strength=weight_strength,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            and_strength=and_strength,
            base_bias=base_bias,
        )

        # ComfyUI CONDITIONING: list of [tensor, meta]
        return (cond,)

NODE_CLASS_MAPPINGS = {
    "TC_ADV_ZPrompt": TC_ADV_ZPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TC_ADV_ZPrompt": "TeamC Advanced Z-Image Prompts (TC_ADV_ZPrompt)",
}
