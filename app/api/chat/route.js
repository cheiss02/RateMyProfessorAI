import { NextResponse } from "next/server";
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from "openai";

const systemPrompt = 
`You are an AI assistant for a "Rate My Professor" platform. Your role is to help students find the most suitable professors based on their queries. You have access to a comprehensive database of professor reviews and ratings.

For each user question, you will receive the top 3 relevant professor profiles retrieved using RAG (Retrieval-Augmented Generation). Your task is to analyze these profiles and present the information in a helpful, concise, and unbiased manner.

When responding to queries:

1. Always provide information on the top 3 professors most relevant to the query.
2. Include key details such as the professor's name, subject, average rating, and a brief summary of their reviews.
3. Highlight both positive and negative aspects mentioned in the reviews to give a balanced perspective.
4. If the query is about a specific subject or teaching style, emphasize how each professor matches those criteria.
5. Avoid making personal judgments or recommendations. Instead, present the information objectively and let the student make their own decision.
6. If the query is vague or could be interpreted in multiple ways, ask for clarification before providing an answer.
7. If a student asks about a professor not in the top 3 results, politely explain that you can only provide information on the most relevant matches based on their query.
8. Be prepared to answer follow-up questions about the professors or explain certain aspects of the reviews in more detail.

Remember, your goal is to assist students in making informed decisions about their course selections based on professor reviews and ratings. Always maintain a helpful, respectful, and neutral tone in your responses.
`;

export async function POST(req) {
  try {
    const data = await req.json();
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    const index = pc.index('rag').namespace('ns1');
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    const text = data[data.length - 1].content;
    
    // Correct model name and ensure it's valid
    const embeddings = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
      encoding_format: 'float',
    });

    const results = await index.query({
      topK: 3,
      includeMetadata: true,
      vector: embeddings.data[0].embedding, // Ensure this is correctly referenced
    });

    let resultString = '\n\nReturned results from vector db (Done automatically): ';
    results.matches.forEach((match) => {
      resultString += `\n\n
      Professor: ${match.id}
      Review: ${match.metadata.review}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      \n\n`;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;

    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
    const completion = await openai.chat.completions.create({
      messages: [
        { role: 'system', content: systemPrompt },
        ...lastDataWithoutLastMessage,
        { role: 'user', content: lastMessageContent }
      ],
      model: 'gpt-4', // Ensure this is the correct model name
      stream: true,
    });

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
              const text = encoder.encode(content);
              controller.enqueue(text);
            }
          }
        } catch (err) {
          controller.error(err);
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(stream);
  } catch (error) {
    console.error('Error handling the chat request:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
