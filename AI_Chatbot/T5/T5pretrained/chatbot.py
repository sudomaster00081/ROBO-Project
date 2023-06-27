import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

config = T5Config(max_length=100)

tokenizer_path = "/home/christy/Documents/Myself/Academia/PG docs/Projects/FR-Robot/AI_Chatbot/T5/T5pretrained/tokenizer"
# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("/home/christy/Documents/Myself/Academia/PG docs/Projects/FR-Robot/AI_Chatbot/T5/T5pretrained/chatbot_model", config=config)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, config=config)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Chatbot function
def chatbot(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate the chatbot response
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=100)

    # Decode the response tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Chatting loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    response = chatbot(user_input)
    print("ChatBot:", response)
