import os
import re
from time import time, sleep
from uuid import uuid4
import openai
import pinecone
from util import *

class Chatbot:
    def __init__(self, name: str, conversation_length: int = 30) -> None:
        self._name = name
        self._conversation_length = conversation_length

        openai.api_key = open_file('key_openai.txt')
        pinecone.init(api_key=open_file('key_pinecone.txt'), environment='us-east1-gcp')

        self._vdb = pinecone.Index("test-chatbot")

    def gpt3_embedding(self, content, engine='text-embedding-ada-002'):
        content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
        response = openai.Embedding.create(input=content,engine=engine)
        vector = response['data'][0]['embedding']  # this is a normal list
        return vector

    def gpt3_completion(self, prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'ENDER:']):
        max_retry = 5
        retry = 0
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        while True:
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=top_p,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stop=stop)
                text = response['choices'][0]['text'].strip()
                text = re.sub('[\r\n]+', '\n', text)
                text = re.sub('[\t ]+', ' ', text)
                filename = '%s_gpt3.txt' % time()
                if not os.path.exists('gpt3_logs'):
                    os.makedirs('gpt3_logs')
                save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
                return text
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "GPT3 error: %s" % oops
                print('Error communicating with OpenAI:', oops)
                sleep(1)

    def load_conversation(self, results):
        result = list()
        for m in results['matches']:
            info = load_json('nexus/%s.json' % m['id'])
            result.append(info)
        ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
        messages = [i['message'] for i in ordered]
        return '\n'.join(messages).strip()
        
    def generate_nexus(self, speaker: str, message: str) -> str:
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        unique_id = str(uuid4())
        metadata = {'speaker': speaker, 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        vector = self.gpt3_embedding(message)
        return unique_id, vector

    def ask(self, message: str):
        payload = list()

        # generate an embedding for the input message and save it
        unique_id, vector = self.generate_nexus('USER', message)
        payload.append((unique_id, vector))

        # search for relevant messages and load from nexus files
        results = self._vdb.query(vector=vector, top_k=self._conversation_length)
        conversation = self.load_conversation(results)  
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', message)

        # generate response, vectorize, save, etc
        chatbot_response = self.gpt3_completion(prompt, stop=['USER:', f'{self._name}:'])

        unique_id, vector = self.generate_nexus(self._name, chatbot_response)
        payload.append((unique_id, vector))

        self._vdb.upsert(payload)

        return chatbot_response

if __name__ == '__main__':
    chatbot = Chatbot("ENDER", 30)

    while True:
        message = input('\n\nUSER: ')
        output = chatbot.ask(message)
        print('\n\nENDER: %s' % output) 