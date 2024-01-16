import os
import base64
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import google.generativeai as genai
from openai import OpenAI
from BingImageCreator import ImageGen  # type: ignore
from telebot.types import Message  # type: ignore
from qweatherpyapi.qweather import QWeatherPy, GeoWeather


def make_gemini_client():
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    client = model.start_chat()
    return client


def has_quota(message: Message, bot_name: str) -> bool:
    """
    Check whether the message has a quota.

    a quota: @bot_name quota? or quota? or /quota or /quota@bot

    Returns:
      bool: If it has a quota, return True. Otherwise, return False.
    """
    msg_text: str = message.text.strip()
    # @bot_name quota?
    if msg_text.startswith("@"):
        if not msg_text.startswith(f"@{bot_name} "):
            return False
        msg_text = msg_text[len(bot_name) + 2:]

    start_words = ["quota?", "/quota"]
    prefix = next((w for w in start_words if msg_text.startswith(w)), None)
    if not prefix:
        return False

    s = msg_text[len(prefix):]
    # /quota@bot
    if s.startswith("@"):
        if s != f"@{bot_name}":
            return False

    return True


def extract_prompt(message: Message, bot_name: str) -> Optional[str]:
    """
    This function filters messages for prompts.

    a prompt: start with @bot or 'prompt:' or '/prompt ' or '/prompt@bot'

    Returns:
      str: If it is not a prompt, return None. Otherwise, return the trimmed prefix of the actual prompt.
    """
    msg_text: str = message.text.strip()
    if msg_text.startswith("@"):
        if not msg_text.startswith(f"@{bot_name} "):
            return None
        s = msg_text[len(bot_name) + 2:]
    else:
        start_words = [
            "prompt:",
            "/prompt",
            "prompt_pro:",
            "/prompt_pro",
            "prompt_gem",
            "/prompt_gem",
        ]
        prefix = next((w for w in start_words if msg_text.startswith(w)), None)
        if not prefix:
            return None
        s = msg_text[len(prefix):]
        # If the first word is '@bot_name', remove it as it is considered part of the command when in a group chat.
        if s.startswith("@"):
            if not s.startswith(f"@{bot_name} "):
                return None
        s = " ".join(s.split(" ")[1:])
    return s


def pro_prompt_by_openai(prompt: str, openai_args: dict, client: OpenAI) -> str:
    prompt = f"revise `{prompt}` to a DALL-E prompt"
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], **openai_args
    )
    res = completion.choices[0].message.content.encode("utf8").decode()
    return res


def pro_prompt_by_gemini(prompt: str, client) -> str:
    # TODO fix the type hint
    prompt = f"revise `{prompt}` to a DALL-E prompt, return the content in English only return the scene and detail"
    client.send_message(prompt)
    return client.last.text


def _image_to_data_uri(file_path):
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_image}"


def pro_prompt_by_openai_vision(prompt: str, openai_args: dict, client: OpenAI) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_uri("temp.jpg")},
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()
    res = response["choices"][0]["message"]["content"].encode("utf8").decode()
    prompt = f"{prompt} {res}"
    res = pro_prompt_by_openai(prompt, openai_args, client)
    return res


def pro_prompt_by_gemini_vision(prompt) -> str:
    model = genai.GenerativeModel("gemini-pro-vision")
    image_path = Path("temp.jpg")
    image_data = image_path.read_bytes()
    contents = {
        "parts": [
            {"mime_type": "image/jpeg", "data": image_data},
            {
                "text": f"Describe this picture as detailed as possible in order to dalle-3 painting plus this {prompt}"
            },
        ]
    }
    response = model.generate_content(contents=contents)
    return response.text


def get_quota(bing_image_obj_list: List[ImageGen]) -> List[Tuple[int, int]]:
    return [(index, v.get_limit_left()) for index, v in enumerate(bing_image_obj_list)]


def save_images(i: ImageGen, images: List[str], path: str) -> None:
    # save the images in another thread call
    print("Running save images")
    i.save_images(images, path)


def prepare_save_images(message: Message) -> str:
    # Prepare the local folder
    print(f"Message from user id {message.from_user.id}")
    path = os.path.join("tg_images", str(message.from_user.id))
    if not os.path.exists(path):
        os.mkdir(path)
    return path


moon = {"新月": "🌑", "蛾眉月": "🌒", "上弦月": "🌓", "盈凸月": "🌔", "满月": "🌕", "亏凸月": "🌖", "下弦月": "🌗",
        "残月": "🌘"}

moon_pic = {"800": "🌑", "801": "🌒", "802": "🌓", "803": "🌔", "804": "🌕", "805": "🌖", "806": "🌗",
            "807": "🌘"}


class Weather:
    def __init__(self, fx_date, sunrise, sunset, temp_max, temp_min, text_day, humidity, uv_index, wind_dir_day,
                 wind_scale_day, wind_speed_day, vis, moon_phase, moon_phase_icon):
        self.fx_date = fx_date
        self.sunrise = sunrise
        self.sunset = sunset
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.text_day = text_day
        self.humidity = humidity
        self.uv_index = uv_index
        self.wind_dir_day = wind_dir_day
        self.wind_scale_day = wind_scale_day
        self.wind_speed_day = wind_speed_day
        self.vis = vis
        self.moon_phase = moon_phase
        self.moon_phase_icon = moon_phase_icon


def get_weather(message: Message, bot_name: str, key: str) -> Optional[Weather]:
    msg_text: str = message.text.strip()
    if msg_text.startswith("@"):
        if not msg_text.startswith(f"@{bot_name} "):
            return None
        s = msg_text[len(bot_name) + 2:]
    else:
        start_words = [
            "weather:",
            "weather_gem:",
            "/weather",
            "/weather_gem",
        ]
        prefix = next((w for w in start_words if msg_text.startswith(w)), None)
        if not prefix:
            return None
        s = msg_text[len(prefix):]
        # If the first word is '@bot_name', remove it as it is considered part of the command when in a group chat.
        if s.startswith("@"):
            if not s.startswith(f"@{bot_name} "):
                return None
        s = " ".join(s.split(" ")[1:])

    city = GeoWeather(key).get_city_lookup(s if s else "shenzhen")
    if city.code != '200':
        return None
    local_city = city.location[0].id

    now_weather = QWeatherPy(key).get_weather_3d(local_city)
    daily = now_weather.daily[0]
    keys = ["fx_date", "sunrise", "sunset", "temp_max", "temp_min", "text_day", "humidity", "uv_index",
            "wind_dir_day", "wind_scale_day", "wind_speed_day", "vis", "moon_phase"]
    data = {key: getattr(daily, key) for key in keys}
    data['moon_phase_icon'] = moon_pic.get(daily.moon_phase_icon)
    return Weather(**data)

def get_msg(weather: Weather) -> str:
    a = f"日期：{weather.fx_date} \n天气情况：{weather.text_day}\n最高温度：{weather.temp_max}\n最低温度：{weather.temp_min}\n风向：{weather.wind_dir_day}\n风力等级：{weather.wind_scale_day}\n风速：{weather.wind_speed_day}\n能见度：{weather.vis}\n湿度：{weather.humidity}%\n紫外线指数：{weather.uv_index}\n日出时间：{weather.sunrise} AM\n日落时间：{weather.sunset} PM\n月相名称：{weather.moon_phase} {weather.moon_phase_icon}"
    return a
