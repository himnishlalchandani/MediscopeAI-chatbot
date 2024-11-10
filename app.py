# app.py
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load FLAN-T5-Large model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_response(user_input, max_length=128):
    # Prepare the prompt
    prompt = f"Given the question or statement: {user_input}\nProvide a helpful and informative response:"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generate response
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,
        length_penalty=1.0,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        num_return_sequences=1,
        early_stopping=True
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.strip()

# Store conversation history
conversation_history = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]
    session_id = request.form.get("session_id", "default")

    # Initialize conversation history for new sessions
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Add user message to history
    conversation_history[session_id].append({"role": "user", "content": user_message})

    try:
        # Generate response
        response = generate_response(user_message)

        # Add response to history
        conversation_history[session_id].append({"role": "assistant", "content": response})

        # Limit history size (optional)
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]

        return jsonify({
            "response": response,
            "success": True
        })

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({
            "response": "I apologize, but I encountered an error processing your message. Please try again.",
            "success": False
        })

@app.route("/clear", methods=["POST"])
def clear_history():
    session_id = request.form.get("session_id", "default")
    if session_id in conversation_history:
        conversation_history[session_id] = []
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)