# from src.tools.tools import web_search,scrape_url

# # output=web_search("Who is ceo of openai?")

# # print(output)

# # res=scrape_url("https://en.wikipedia.org/wiki/Bugatti")

# # print(res)

# res=web_search.invoke("What is latest research in AGI?")

from src.pipelines.pipeline import research_pipeline

topic="impact of ai on job market"

print(research_pipeline(topic=topic))