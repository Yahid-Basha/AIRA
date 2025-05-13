from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
import asyncio

async def search_old_async(client, date, collection_name, query, payload=[], encoder=SentenceTransformer("all-MiniLM-L6-v2"), limit=15):
    try:
        # Verify if the collection exists
        collections = await client.get_collections()
        if not any(collection.name == collection_name for collection in collections.collections):
            return {"status": "error", "message": f"Collection '{collection_name}' does not exist."}

        # Query Qdrant
        hits = await client.search(
            collection_name=collection_name,
            query_vector=encoder.encode(query).tolist(),
            query_filter=models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="Opened",
                        range=models.DatetimeRange(gte=date)
                    ),
                    models.IsNullCondition(is_null=models.PayloadField(key="Closed")),
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="Closed"))
                ]
            ),
            with_payload=["Number", "Description", "Priority", "Opened", "Closed", 
                         "Resolution Code", "Resolution Notes (Internal)", "Resolution Update"],
            limit=limit
        )

        # Process results
        past_tickets = [(hit.payload, hit.score) for hit in hits]
        return past_tickets

    except Exception as e:
        print(f"Error in search_old: {e}")
        return []

async def get_resolution_async(client, collection_name, ticket_description, payload=[], limit=10):
    collection_name = f"{collection_name.upper()}_Resolutions"
    try:
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        hits = await client.search(
            collection_name=collection_name,
            query_vector=encoder.encode(ticket_description).tolist(),
            with_payload=True if payload == [] else payload,
            limit=limit
        )

        resolutions = [hit.payload for hit in hits]
        return resolutions

    except Exception as e:
        print(f"Error fetching resolution: {e}")
        return []
    


async def process_tickets_async(tickets_fetched, client, start_date, collection_name):
    # Create tasks for all tickets
    tasks = []
    for ticket in tickets_fetched:
        # Create tasks for each ticket's operations
        similar_tickets_task = search_old_async(
            client, start_date, collection_name, ticket["Description"], 
            payload=["Number", "Description", "Priority", "Opened", "Closed", 
                        "Resolution Code", "Resolution Notes (Internal)", "Resolution Update"], 
            limit=7
        )
        
        resolution_task = get_resolution_async(
            client=client, 
            collection_name=collection_name, 
            ticket_description=ticket["Description"], 
            payload=[], 
            limit=2
        )
        
        # Add both tasks to the list
        tasks.append((ticket, similar_tickets_task, resolution_task))

    # Process all tickets concurrently
    combined_tickets_data = []
    for ticket, similar_tickets_task, resolution_task in tasks:
        # Wait for both operations to complete
        similar_past_tickets, resolution_steps = await asyncio.gather(
            similar_tickets_task, 
            resolution_task
        )
        
        combined_ticket_data = {
            ticket["Number"]: {
                "ticket_data": ticket,
                "similar_past_tickets": similar_past_tickets,
                "resolution_steps": resolution_steps
            }
        }
        combined_tickets_data.append(combined_ticket_data)

    return combined_tickets_data




@cl.on_message
async def main(message: cl.Message):
    # Initialize async client
    client = AsyncQdrantClient("localhost", port=6333)
    
    # Your existing code to get tickets_fetched...
    
    # Process tickets concurrently
    combined_tickets_data = await process_tickets_async(
        tickets_fetched, 
        client, 
        start_date, 
        collection_name
    )
    
    # Rest of your code...
