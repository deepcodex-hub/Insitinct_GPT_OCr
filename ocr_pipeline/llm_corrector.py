import json
import re
import os

class LLMCorrector:
    """Prompt wrapper for an LLM (Phi-2 / Mistral / Gemma) to perform contextual correction."""
    def __init__(self, endpoint_url="http://localhost:8000/v1/completions", model_name="phi-2"):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        
    def generate_prompt(self, raw_text: str, schema_regex: str, substitutions: dict):
        sys_prompt = (
            "You are an expert OCR correction system. Your task is to correct OCR output for utility meters.\n"
            "Format Regex: {regex}\n"
            "Common substitutions: {subs}\n\n"
            "You must output ONLY valid JSON in the exact schema:\n"
            "{{\n"
            '  "best": "string",\n'
            '  "alts": [{{"text": "string", "score": float}}],\n'
            '  "reasons": "string"\n'
            "}}\n"
        ).format(regex=schema_regex, subs=json.dumps(substitutions))
        
        prompt = f"{sys_prompt}\nRaw OCR Text:\n{raw_text}\nJSON Output:\n"
        return prompt

    def parse_output(self, llm_response: str, fallback: str):
        """Safely parses LLM output enforcing schema constraints."""
        try:
            # simple extract json blocks
            json_block = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_block:
                data = json.loads(json_block.group(0))
                # Validate schema loosely
                if 'best' in data:
                    return data
        except Exception:
            pass
        
        # Fallback if LLM hallucinations
        return {
            "best": fallback,
            "alts": [],
            "reasons": "LLM failed to output parseable JSON. Substituted fallback."
        }
        
    def correct(self, raw_text: str, regex=r'^\d{5,8}$'):
        substitutions = {
            'O': '0', 'o': '0', 'I': '1', 'l': '1', '|': '1',
            'S': '5', 's': '5', 'B': '8', 'Z': '2', 'z': '2'
        }
        prompt = self.generate_prompt(raw_text, regex, substitutions)
        
        # Stubbed HTTP call to a local quantized LLM (e.g., via vLLM or text-generation-webui)
        # requests.post(...)
        
        # Mock LLM Response logic for MVP demonstration
        import re as regex_mod
        cleaned = raw_text
        for k, v in substitutions.items(): cleaned = cleaned.replace(k, v)
        cleaned = regex_mod.sub(r'[^0-9.]', '', cleaned)
        
        mock_response = json.dumps({
            "best": cleaned if cleaned else raw_text,
            "alts": [{"text": raw_text, "score": 0.5}],
            "reasons": "Mock LLM applied basic deterministic substitutions."
        })
        
        return self.parse_output(mock_response, raw_text)
