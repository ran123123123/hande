import os
import os.path as osp
import json
import cv2
from pydub import AudioSegment

import const
import cv_task.func as cv_task
import nlp_task.func as nlp_task
import speech_task.func as speech_task

def test_cv_task(imga_path, imgb_path, output_path):
    config = json.loads(const.EV_AUTO_TEST_CONFIG_PARAMS)
    config["data_set_type"] = "CV"
    config["pretreatment"] = {
        "fill_flag": 1,
        "fill_args": {"roi": [100, 100, 200, 200]},

        "hist_equa_flag": 1,
        "white_balance_flag": 1,
        "automatic_color_enhancement_flag": 1,
        "blur_flag": 1 
    }
    config["save_result"] = False
    ret_json = json.loads(const.RET_JSON)

    imga = cv2.imread(imga_path, -1)
    imgb = cv2.imread(imgb_path, -1)
    # save_img = "result"

    # No.1
    roi = config["pretreatment"]["fill_args"]["roi"]
    if config["pretreatment"]["fill_flag"]:
        imga = cv_task.fill(imga, imgb, roi)
        if config["save_result"]:
            save_img = f"result_fill.jpeg"
            cv2.imwrite(save_img, imga)

    # No.2
    if config["pretreatment"]["hist_equa_flag"]:
        imga = cv_task.hist_equa(imga)
        if config["save_result"]:
            save_img = f"result_hist_equa.jpeg"
            cv2.imwrite(save_img, imga)

    # No.3
    if config["pretreatment"]["white_balance_flag"]:
        imga = cv_task.white_balance(imga)
        if config["save_result"]:
            save_img = f"result_white_balance.jpeg"
            cv2.imwrite(save_img, imga)

    # No.4
    if config["pretreatment"]["automatic_color_enhancement_flag"]:
        imga = cv_task.automatic_color_enhancement(imga)
        if config["save_result"]:
            save_img = f"result_automatic_color_enhancement.jpeg"
            cv2.imwrite(save_img, imga)

    # No.5
    if config["pretreatment"]["blur_flag"]:
        imga = cv_task.blur(imga)
        if config["save_result"]:
            save_img = f"result_blur.jpeg"
            cv2.imwrite(save_img, imga)

    cv2.imwrite(output_path, imga)
    ret_json["data"]["output_path"] = output_path
    return ret_json

def test_nlp_task(input_str):
    config = json.loads(const.EV_AUTO_TEST_CONFIG_PARAMS)
    config["data_set_type"] = "NLP"
    config["pretreatment"] = {
        "remove_number_flag": 1,
        "remove_space_flag": 1,
        "remove_url_flag": 1,
        "remove_duplicate_str_flag": 1 
    }
    config["show_result"] = True
    ret_json = json.loads(const.RET_JSON)

    # No.3
    if config["pretreatment"]["remove_url_flag"]:
        input_str = nlp_task.remove_url(input_str)
        if config["show_result"]:
            print(f"after remove_url: {input_str}")

    # No.1
    if config["pretreatment"]["remove_number_flag"]:
        input_str = nlp_task.remove_number(input_str)
        if config["show_result"]:
            print(f"after remove_number: {input_str}")

    # No.2
    if config["pretreatment"]["remove_space_flag"]:
        input_str = nlp_task.remove_space(input_str)
        if config["show_result"]:
            print(f"after remove_space: {input_str}")

    # No.4
    input_str_list = [input_str] * 10
    if config["pretreatment"]["remove_duplicate_str_flag"]:
        input_str_list = nlp_task.remove_duplicate_str(input_str_list)
        if config["show_result"]:
            print(f"after remove_duplicate_str: {input_str_list}")

    ret_json["data"]["output_str"] = input_str_list
    return ret_json

def test_speech_task(wave_file, output_path):
    config = json.loads(const.EV_AUTO_TEST_CONFIG_PARAMS)
    config["data_set_type"] = "SPEECH"
    config["pretreatment"] = {
        "denoise_flag": 1,
        "remove_silence_flag": 1,
        "increase_sound_flag": 1,
        "increase_sound_args": {"inc": 10}
    }
    config["save_result"] = False
    ret_json = json.loads(const.RET_JSON)

    wave_data = AudioSegment.from_file(wave_file)

    # No.1
    if config["pretreatment"]["denoise_flag"]:
        wave_data = speech_task.denoise(wave_data)
        if config["save_result"]:
            save_wav = f"result_denoise.wav"
            wave_data.export(save_wav, format="wav")

    # No.2
    if config["pretreatment"]["remove_silence_flag"]:
        wave_data = speech_task.remove_silence(wave_data)
        if config["save_result"]:
            save_wav = f"result_remove_silence.wav"
            wave_data.export(save_wav, format="wav")

    # No.3
    inc = config["pretreatment"]["increase_sound_args"]["inc"]
    if config["pretreatment"]["increase_sound_flag"]:
        wave_data = speech_task.increase_sound(wave_data, inc)
        if config["save_result"]:
            save_wav = f"result_increase_sound.wav"
            wave_data.export(save_wav, format="wav")

    wave_data.export(output_path, format="wav")
    ret_json["data"]["output_path"] = output_path
    return ret_json

if __name__ == "__main__":
    # for CV
    imga, imgb, output_path = "cv_task/test.jpeg", "cv_task/test.jpeg", "task_result.jpg"
    ret_json = test_cv_task(imga, imgb, output_path)
    with open("test_cv_task_result.json", "w", encoding="utf-8") as fout:
        json.dump(ret_json, fout, indent=2)

    # for NLP
    # 第四个函数做文本去重需要用到字符串数组，实现里面将输入字符串copy了10份.
    input_str = "xxxx xxxx 123 https://baidu.com yyyy"
    ret_json = test_nlp_task(input_str)
    with open("test_nlp_task_result.json", "w", encoding="utf-8") as fout:
        json.dump(ret_json, fout, indent=2)

    # for SPEECH
    wave_file, output_path = "speech_task/test.wav", "task_result.wav"
    ret_json = test_speech_task(wave_file, output_path)
    with open("test_speech_task_result.json", "w", encoding="utf-8") as fout:
        json.dump(ret_json, fout, indent=2)
