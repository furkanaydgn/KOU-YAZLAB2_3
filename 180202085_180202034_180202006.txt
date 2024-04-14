from tkinter.ttk import Entry
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from tkinter import Tk, Frame, Button, Label, Text, filedialog, StringVar
import networkx as nx
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import torch
from tkinter import OptionMenu
from rouge import Rouge

sentences = []
sentence_embeddings = {}
graph = {}
title_words = set()
threshold = 0.5
reference_text = ""
rouge_scores = ""

root = Tk()
root.title("Document Summarization Tool")
root.geometry("800x600")

def preprocess_text(content):
    # Tokenization
    sentences = sent_tokenize(content)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    processed_sentences = []

    for sentence in sentences:
        # Remove punctuation marks
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Tokenization and stemming
        words = [stemmer.stem(word) for word in word_tokenize(sentence.lower()) if word not in stop_words]
        processed_sentences.append(' '.join(words))

    return processed_sentences



def calculate_sentence_scores(sentences,title,content):
    scores = []
    original_sentences = []
    theme_words = set()
    total_words = 0


    def calculate_p1(sentence):
        tagged_words = nltk.pos_tag(word_tokenize(sentence))
        proper_nouns = [word for word, pos in tagged_words if pos == 'NNP']
        return len(proper_nouns) / len(sentence.split())


    def calculate_p2(sentence):
        numerical_data = sum([1 for word in word_tokenize(sentence) if word.isdigit()])
        return numerical_data / len(sentence.split())


    def calculate_p3(sentence, connections, similarity_threshold):
        similar_connections = sum([1 for connection in connections if cosine_similarity([sentence_embeddings[sentence]],
                                                                                        [sentence_embeddings[
                                                                                             connection]]) >= similarity_threshold])
        return similar_connections / len(connections)

    # Checking whether there are words in the title in the sentence (P4)
    def calculate_p4(sentence, title_words):
        title_count = sum([1 for word in word_tokenize(sentence) if word in title_words])
        return title_count / len(sentence.split())

    def calculate_p5(sentence, tfidf_matrix, feature_names, theme_words_count):
        sentence_words = sentence.split()
        theme_words = set(feature_names[:theme_words_count])

        theme_word_count = sum(1 for word in sentence_words if word in theme_words)
        return theme_word_count / len(sentence_words)

    # Determine theme words for TF-IDF calculation
    for sentence in sentences:
        total_words += len(sentence.split())
    theme_words_count = int(total_words * 0.1)

    # Calculate TF-IDF values
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    theme_words_count = int(len(sentences) * 0.1)
    threshold_skor = float(threshold_score_entry.get())

    for i,sentence in enumerate(sentences):
        original_sentence = content.split('.')[i].strip()  # Get the corresponding original unprocessed sentence
        original_sentences.append(original_sentence)  # Store the original unprocessed sentence
        p1 = calculate_p1(sentence)
        p2 = calculate_p2(sentence)
        p3 = calculate_p3(sentence, sentences,threshold_skor)  # Adjust the similarity threshold as desired
        p4 = calculate_p4(sentence, title)  # Replace 'title_words' with the words from the document title
        p5 = calculate_p5(sentence, tfidf_matrix, feature_names, theme_words_count)

        score = (p1 + p2 + p3 + p4 + p5) / 5.0
        scores.append(score)



    summary = []

    sorted_scores, sorted_original_sentences = zip(*sorted(zip(scores, original_sentences), reverse=True))
    for sentence in sorted_original_sentences[:7]:
        original_sentence = sentence.strip('{}')
        print("Original Sentence:", original_sentence)
        summary.append(original_sentence.strip('{}'))


    return scores ,summary

def get_title(paragraph):
    lines = paragraph.split('\n')  # Paragrafı satırlara bölmek
    first_line = lines[0].strip()  # İlk satırı almak ve gereksiz boşlukları temizlemek
    return nltk.sent_tokenize(first_line)


def bert_method(sentences):
    sentence_embeddings = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            outputs = model(input_ids)
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

        sentence_embeddings[sentence] = sentence_embedding

    return sentence_embeddings



def word_embedding(sentences):

    model = Word2Vec(sentences=[sentence.lower().split() for sentence in sentences], min_count=1, vector_size=300)

    # Map each sentence to its corresponding embedding
    sentence_embeddings = {}
    for sentence in sentences:
        words = sentence.split()
        embedding = []
        for word in words:
            if word in model.wv:
                # Get the word embedding for the current word
                word_embedding = model.wv[word]
                embedding.append(word_embedding)
        if embedding:
            # Calculate the average embedding for the sentence
            sentence_embedding = sum(embedding) / len(embedding)
            sentence_embeddings[sentence] = sentence_embedding

    return sentence_embeddings





def load_document():
    global sentences, sentence_embeddings, graph, ranked_sentences, summary

    filename = filedialog.askopenfilename(initialdir="/", title="Select Document",
                                          filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
    if filename:
        with open(filename, 'r') as file:
            lines = file.readlines()

        content = ''.join(lines[1:]).strip()
        title = lines[0].strip()
        title_words = set(word_tokenize(title.lower()))
        sentences = preprocess_text(content)


        if word_embedding_var.get() == 'Word Embedding':
            sentence_embeddings = word_embedding(sentences)
        else:
            sentence_embeddings = bert_method(sentences)

        scores, summary = calculate_sentence_scores(sentences, title_words,content)

        # Update the summary display
        summary_text.delete("1.0", "end")
        summary_text.insert("end", summary)
        # Update threshold value from user input
        threshold = float(threshold_entry.get())


        graph = {sentence: [] for sentence in sentences}
        edge_weights = {}
        for sentence1 in sentences:
            for sentence2 in sentences:
                if sentence1 != sentence2:
                    similarity = cosine_similarity([sentence_embeddings[sentence1]], [sentence_embeddings[sentence2]])
                    if similarity >= threshold:  # Adjust the similarity threshold as desired
                        graph[sentence1].append(sentence2)

                        edge_weights[(sentence1, sentence2)] = similarity

        # Visualize the graph
        G = nx.DiGraph(graph)
        pos = nx.spring_layout(G)

        # Add all nodes to pos dictionary
        for node in G.nodes:
            pos[node] = pos.get(node, (0, 0))

        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=900, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0)


        # Draw node labels with sentence content and scores
        node_labels = {node: f"Cumle:{sentences.index(node)}\nScore: {scores[sentences.index(node)]:.2f}" for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8,
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        # Düğüm etiketlerini ekleme
        node_labels_2 = {node: f" {len(graph[node])}" for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels_2, font_size=4,
                                bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.2'))

        for edge in G.edges:
            x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
            y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
            weight = float(edge_weights.get(edge, 0))  # float olarak dönüştürme
            plt.text(x, y, f" {weight:.2f}", fontsize=8, color='red',
                     horizontalalignment='center', verticalalignment='center')

        plt.title("Graph Visualization")
        plt.axis('off')
        plt.show()

def calculate_rouge_score():
    global rouge_scores

    reference = reference_text.get("1.0", "end").strip().lower()
    summary = summary_text.get("1.0", "end").strip().lower()

    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)[0]

    rouge_scores = f"ROUGE-1: {scores['rouge-1']['f']:.2f}, ROUGE-2: {scores['rouge-2']['f']:.2f}, ROUGE-L: {scores['rouge-l']['f']:.2f}"

    rouge_label.config(text=rouge_scores)


word_embedding_var = StringVar()
word_embedding_var.set('Word Embedding')

word_embedding_menu = OptionMenu(root, word_embedding_var, 'Word Embedding', 'BERT Method')
word_embedding_menu.pack()

file_button = Button(root, text="Load Document", command=load_document)
file_button.pack()


threshold_label = Label(root, text="Cümle Benzerliği Thresholdu:")
threshold_label.pack()

threshold_entry = Entry(root)
threshold_entry.pack()

threshold_score_label = Label(root, text="Cümle Skor Thresholdu:")
threshold_score_label.pack()

threshold_score_entry = Entry(root)
threshold_score_entry.pack()

summary_label = Label(root, text="Summary:")
summary_label.pack()

summary_text = Text(root, height=10)
summary_text.pack()

reference_label = Label(root, text="Reference Text:")
reference_label.pack()

reference_text = Text(root, height=6)
reference_text.pack()

rouge_button = Button(root, text="Calculate Rouge Score", command=calculate_rouge_score)
rouge_button.pack()

rouge_label = Label(root, text="ROUGE Scores:")
rouge_label.pack()

graph_frame = Frame(root)
graph_frame.pack()

root.mainloop()