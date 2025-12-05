# import os
# from dotenv import load_dotenv
# load_dotenv()
# from openai import AsyncOpenAI

# client = AsyncOpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://genailab.tcs.in"
# )

# # Define an async function to use await
# async def main():
#     # Use chat.completions.create for conversational AI
#     response = await client.chat.completions.create(
#         model="azure_ai/genailab-maas-DeepSeek-V3-0324",
#         messages=[
#             {"role": "user", "content": "How does AI work?"}
#         ],
#         temperature=0.7,
#         max_tokens=500,
#     )
    
#     # Extract the generated text from the response
#     print(response.choices[0].message.content)

# # Run the async function
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class DeepSeekGenerator:
    def __init__(self, model="azure_ai/genailab-maas-DeepSeek-V3-0324"):
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://genailab.tcs.in"
        )
        self.model = model
        
    async def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """
        Generate response from DeepSeek-V3
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Corrected API call for AsyncOpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
                stream=False
            )
            
            # Properly extract the generated content
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API error: {str(e)}")
            raise ValueError("Failed to generate response from DeepSeek-V3") from e
async def main():
    generator = DeepSeekGenerator()
    response = await generator.generate(
        prompt="Explain quantum computing in simple terms",
        system_prompt="You are a helpful AI assistant."
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())