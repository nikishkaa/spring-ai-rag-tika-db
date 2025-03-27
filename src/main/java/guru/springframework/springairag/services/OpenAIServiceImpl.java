package guru.springframework.springairag.services;

import guru.springframework.springairag.model.Answer;
import guru.springframework.springairag.model.Question;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * Created by jt, Spring Framework Guru.
 */
@Service
public class OpenAIServiceImpl implements OpenAIService {

    private final ChatModel chatModel;
    private final SimpleVectorStore vectorStore;

    @Value("classpath:/templates/cayen-pompt-template.st")
    private Resource ragPromptTemplate;

    public OpenAIServiceImpl(ChatModel chatModel, SimpleVectorStore vectorStore) {
        this.chatModel = chatModel;
        this.vectorStore = vectorStore;
    }

    @Override
    public Answer getAnswer(Question question) {

        List<Document> documents = vectorStore.similaritySearch(SearchRequest.builder()
                .query(question.question()).topK(10).build());
        List<String> contentList = documents.stream().map(Document::getContent).toList();

        PromptTemplate promptTemplate = new PromptTemplate(ragPromptTemplate);
        Prompt prompt = promptTemplate.create(Map.of("input", question.question(), "documents",
                String.join("\n", contentList)));

        contentList.forEach(System.out::println);

        ChatResponse response = chatModel.call(prompt);

        return new Answer(response.getResult().getOutput().getContent());
    }
}
