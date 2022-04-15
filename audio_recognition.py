#!/usr/bin/env python
import rospy
from audio_common_msgs.msg import AudioData
import torch
import numpy as np
from queue import  Queue
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import threading
import time

torch.random.manual_seed(47)

class AudioInference():
    def __init__(self,model_name):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) 
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def buffer_to_text(self,audio_buffer):
        if(len(audio_buffer)==0):
            print("return mevzusu")
            return ""
        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)        
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()


class audio_recognition():
    exit_event = threading.Event()   
    def __init__(self, iscuda : bool, sample_rate : int, iswav : bool, model_name : str):
        self.audio_frames = b''
        self.iscuda = iscuda
        self.sample_rate = sample_rate
        self.iswav = iswav
        self.model_name=model_name
        rospy.init_node('listener', anonymous=True)
        self.device = torch.device("cuda" if self.iscuda else "cpu")
        self.wave2vec_asr = AudioInference(self.model_name)
        self.output=["",3,1]

    def audio_inference(self,data):
        if len(list(bytearray(self.audio_frames))) > 320000:
            text,sample_length,inference_time = self.asr_process(self.audio_frames)                       
            print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{text}")
            self.audio_frames = b''
        self.audio_frames += data.data
        #print(len(list(bytearray(self.audio_frames))))

    def asr_process(self, audio_frames):

        
        print("debug")               


        float64_buffer = np.frombuffer(
            audio_frames, dtype=np.int16) / 32767

        start = time.perf_counter()
        print(float64_buffer)
        text = self.wave2vec_asr.buffer_to_text(float64_buffer).lower()
        inference_time = time.perf_counter()-start
        sample_length = len(float64_buffer) / 16000  # length in sec

        output=[text,sample_length,inference_time]    
        return output


if __name__ == '__main__':
    model_name= "facebook/wav2vec2-large-960h-lv60-self" #Ingilizce egitilmis model
    #model_name="m3hrdadfi/wav2vec2-large-xlsr-turkish" #Turkce egitilmis model
    recog = audio_recognition(True, 16000,True,model_name)

    rospy.Subscriber("audio", AudioData, recog.audio_inference)


    rospy.spin()
