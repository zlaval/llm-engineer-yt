import json

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai = OpenAI()

car_prices = {
    "toyota": "24000",
    "honda": "26000",
    "ford": "32000",
    "chevrolet": "23000",
    "bmw": "41000",
    "tesla": "39000"
}


def get_price(car_model):
    car = car_model.lower()
    return car_prices.get(car, "Nincs ilyen")


price_fn = {
    "name": "get_price",
    "description": "Get the price of a car",
    "parameters": {
        "type": "object",
        "properties": {
            "car_model": {
                "type": "string",
                "description": "The car model",
            },
        },
        "required": ["car_model"],
        "additionalProperties": False,
    },
}

tools = [{
    "type": "function",
    "function": price_fn
}]


def chat(prompt, history):
    message = [
        {
            "role": "system",
            "content": """You are an polite assistant of a car seller company called CarComp.
             Always be accurate, if you dont know the answer, say 'Nem tudom'.
             Give short, exact answers in one sentence.
             """,
        },
    ]

    for um, am in history:
        message.append({"role": "user", "content": um})
        message.append({"role": "assistant", "content": am})

    message.append({"role": "user", "content": prompt})

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=message,
        tools=tools
    )

    if response.choices[0].finish_reason == "tool_calls":
        msg = response.choices[0].message
        resp, price = handle_tool_call(msg)
        message.append(msg)
        message.append(resp)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=message
        )

    return response.choices[0].message.content or ""


def handle_tool_call(msg):
    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    car = args.get("car_model")
    price = get_price(car)
    res = {
        "role": "tool",
        "content": f"The price of {car} is {price} Eur.",
        "tool_call_id": tool_call.id,
    }
    return res, price


gr.ChatInterface(fn=chat).launch()
