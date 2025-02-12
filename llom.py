# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode the input prompt (text that we want to start generating from)
input_text = "Once upon a time in a land far, far away,"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text from the model
output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
