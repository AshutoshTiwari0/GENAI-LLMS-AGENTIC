#in this file we will connect all the tools, agents together. So import all the tools and agents here


from src.agents.agents import critic_chain,writer_chain,build_reader_agent,build_search_agent

def research_pipeline(topic:str)->str:

    """Main pipeline for researching on the provided topic"""
    state={}  #to store results of agent.

    #search agent working
    print("Search agent is working")

    search_agent=build_search_agent()
    search_result=search_agent.invoke({
        "messages":[("user",f"Find recent information about {topic}")]
    })

    state["search_results"]=search_result['messages'][-1].content

    print("search result is ", state['search_results'])



    #now reader agent works
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nscraped content: \n", state['scraped_content'])



    #now writer chain works
    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])



    #now critic chain works
    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state