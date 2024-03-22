from openai import OpenAI
import sys

if len(sys.argv) != 3:
  print("Usage: python inference.py <vllm server> <model>")
  print("  Example: python inference.py http://localhost:8080/v1 angelmmiguel/phi2-intro-finetuned-demo")
  exit(1)

server_url = sys.argv[1]
model_id = sys.argv[2]

client = OpenAI(
    # This is the default and can be omitted
    api_key="fake-key",
    base_url = server_url
)

# Just for demo purposes...
query = "My name is Karolina Nowak, I'm 35 years old, and I'm a highly motivated and dedicated professional looking for a new challenge in my career. I have a passion for teamwork, problem-solving, and delivering high-quality results. In my free time, I enjoy hiking and playing the guitar. I'm currently based in London, UK, where I work as a Marketing Manager for a tech startup. I'm fluent in Polish, English, and Spanish, and I have a Master's degree in Business Administration from the University of Warsaw. I'm excited to explore new opportunities and connect with like-minded professionals."

print("Extracting data from...")
print(query)

# Final prompt
prompt = f"""
You are a smart assistant that follows instructions:

### Instruction
Extract information from a given text in a structured way. Based on the user message, you must extract the following information: name, age, location and role. Then, you return these 4 properties in the following format:

{{ "name": "NAME", "age": "AGE", "location": "LOCATION", "role": "ROLE" }}

Where you change the uppercased words with the values extracted from the original text. You must return only this data as output, skipping any other text before and after. If there's a double quote symbol in any of the values, escape it using the \ symbol.

Input: {query}
Output:"""

response = client.completions.create(
  model=model_id,
  prompt=prompt,
  max_tokens=100,
  stop=["\n\n"],
)

print("\nResult:")
print(response.choices[0].text.strip())
