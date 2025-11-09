from flask import Flask, render_template, request, jsonify

from bot.chatbot import ChatBot

app = Flask(__name__)

chatbot = ChatBot()


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    user_message = request.json["message"]

    print(f"Received message: {user_message}")  # debug

    response = chatbot.get_answer(user_message)

    print(f"Reply: {response}")  # debug

    return jsonify({"reply": response})


if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=5000)
