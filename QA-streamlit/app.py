import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# -------------------------------
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

st.title("Question Answering with DistilBERT")
st.write("Enter a passage (context) and a question to get the answer.")

# Text area for passage
context = st.text_area("Enter Passage (Context):", height=200)

# Text input for question
question = st.text_input("Enter Question:")

# Button to get answer
if st.button("Get Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Please enter both a passage and a question!")
    else:
        # -------------------------------
        # Step 3: Tokenize and Predict
        # -------------------------------
        inputs = tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract answer span
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1])
        )

        st.subheader("Predicted Answer:")
        st.write(answer)