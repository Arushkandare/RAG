from google import generativeai as genai

genai.configure(api_key="AIzaSyDGiDme3NtO47LmWgqv_Fr7UC908ybYaC0")

response = genai.GenerativeModel("gemini-1.5-flash").generate_content("Explain how LLM works")

print(response.text)