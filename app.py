import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def initialize_model():
    path = "jianghc/medical_chatbot"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path).to(device)
    return tokenizer, model, device

def generate_response(input_text, tokenizer, model, device):
    prompt_input = (
        "The conversation between human and AI assistant.\n"
        "[|Human|] {input}\n"
        "[|AI|]"
    )
    sentence = prompt_input.format(input=input_text)
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        beam_output = model.generate(
            **inputs,
            min_new_tokens=1, 
            max_length=512,
            num_beams=3,
            repetition_penalty=1.2,
            early_stopping=True,
            eos_token_id=198
        )
    return tokenizer.decode(beam_output[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model, device = initialize_model()
    user_input = "what is Parkinson's disease?"
    response = generate_response(user_input, tokenizer, model, device)
    print(response)
