"""
This code is based on https://cloud.google.com/text-to-speech/docs/samples

It implements an audio based chatbot using the chatbot class from chatbot.py
"""

from __future__ import division

import re
import sys
import html
import json
from chatbot import Chatbot

from google.cloud import speech, texttospeech
from google.oauth2 import service_account

import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(ttsclient, responses, chatbot, output_stream):
    """Iterates through server responses and prints them."""

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE
    )
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print("USER: " + transcript + overwrite_chars)
            
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break
            
            chatbot_response = chatbot.ask(transcript) 
            print(f"{chatbot._name}: {chatbot_response}")
        
            input_text = texttospeech.SynthesisInput(text=chatbot_response)

            # Perform the text-to-speech request
            response = ttsclient.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)       
            
            # with open("test.wav", "wb") as out:
            #     out.write(response.audio_content)
        
            output_stream.write(response.audio_content)
            
            num_chars_printed = 0


def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-US"  # a BCP-47 language tag

    credentials = service_account.Credentials.from_service_account_file("key_google.json")

    sttclient = speech.SpeechClient(credentials=credentials)
    ttsclient = texttospeech.TextToSpeechClient(credentials=credentials)
    
    chatbot = Chatbot("ENDER")

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    p = pyaudio.PyAudio()
    output_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, frames_per_buffer=CHUNK, output=True)
    output_stream.start_stream()

    with MicrophoneStream(RATE, CHUNK) as input_stream:
        audio_generator = input_stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = sttclient.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(ttsclient, responses, chatbot, output_stream)


    output_stream.stop_stream()
    output_stream.close()
    p.terminate()

if __name__ == "__main__":
    main()