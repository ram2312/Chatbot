 
def import_packages():

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertModel
    import torch.nn.functional as F

# HR questions and answers
def extract_hr_question_pairs():
    with open('hr_specific_dataset.txt', 'r') as file:
        data = file.readlines()

    questions = []
    answers = []
    for line in data:
        if line.startswith('Q:'):
            questions.append(line.strip().replace('Q: ',''))
        elif line.startswith('A:'):
            answers.append(line.strip().replace('A: ',''))

    questions=questions[0:10]
    answers=answers[0:10]

    print(questions)
    print(answers)

# Define the HR dataset
class HRDataset():
    def __init__(self, questions, answers, tokenizer):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        encoded = self.tokenizer.encode_plus(question, answer, add_special_tokens=True, 
                                              max_length=512, truncation=True, padding='max_length')
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        token_type_ids = torch.tensor(encoded['token_type_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

# Define the BERT-based HR model
class HRModel():
    def __init__(self):
        super(HRModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Train the HR model
def train_hr_model(questions, answers, tokenizer):
    # Define the dataset and data loader
    dataset = HRDataset(questions, answers, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Define the HR model, optimizer, and loss function
    model = HRModel()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train the HR model for 5 epochs
    for epoch in range(5):
        running_loss = 0.0
        for input_ids, token_type_ids, attention_mask in dataloader:
            optimizer.zero_grad()
            logits = model(input_ids, token_type_ids, attention_mask)
            labels = torch.ones_like(logits)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
    
    return model

# Fine-tune the pre-trained BERT model on the HR dataset
def fine_tune_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = train_hr_model(questions, answers, tokenizer)

def SaveModel():
    from transformers import BertForQuestionAnswering, BertTokenizer

    # Load the pre-trained BERT model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Fine-tune the model

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('fine-tuned-hr-bert')
    tokenizer.save_pretrained('fine-tuned-hr-bert')



def LoadModel():
    from transformers import BertForQuestionAnswering, BertTokenizer

    # Load the fine-tuned model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('fine-tuned-hr-bert')
    tokenizer = BertTokenizer.from_pretrained('fine-tuned-hr-bert')
    return model, tokenizer

#model, tokenizer = LoadModel()

# Load pre-trained BERT model and tokenizer
def loader():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ir_model = BertModel.from_pretrained('bert-base-uncased')
    # Encode answers into vectors
    answer_vectors = []
    for answer in answers:
        encoded_answer = tokenizer.encode(answer, add_special_tokens=True)
        input_ids = torch.tensor(encoded_answer).unsqueeze(0)
        outputs = ir_model(input_ids)
        last_hidden_state = outputs[0].squeeze(0)
        answer_vectors.append(last_hidden_state.mean(dim=0).detach().numpy())




# Define function to find the most similar answer to a question
def find_most_similar_answer(question):
    # Encode question into vector
    encoded_question = tokenizer.encode(question, add_special_tokens=True)
    input_ids = torch.tensor(encoded_question).unsqueeze(0)
    outputs = ir_model(input_ids)
    last_hidden_state = outputs[0].squeeze(0)
    question_vector = last_hidden_state.mean(dim=0).detach().numpy()
    
    # Calculate cosine similarities between question vector and answer vectors
    similarities = []
    for answer_vector in answer_vectors:
        similarity = F.cosine_similarity(torch.tensor(question_vector), torch.tensor(answer_vector), dim=0)
        similarities.append(similarity)
    
    # Return answer with highest cosine similarity
    index = torch.argmax(torch.tensor(similarities))
    return answers[index]

def predict_answer(question, model, tokenizer, answers):
    #return find_most_similar_answer(question)
    # Tokenize the question and answers
    inputs = tokenizer.batch_encode_plus([(question, answer) for answer in answers], 
                                          return_tensors='pt',
                                          pad_to_max_length=True,
                                          max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # Get the model's output
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # Find the answer with the highest score
    answer_idx = torch.argmax(output)
    return answers[answer_idx]
def user_input():
    # Test the model on sample questions
    test_questions = ['What is your HR policy on employee benefits and culture?',
                      'How do you ensure a safe and healthy work environment for employees?',
                      'What is your company culture like?',
                      'How do you handle the employee feedback and suggestions in corporation?']
    for question in test_questions:
        #answer = predict_answer(question, model, tokenizer, answers)
        #answer = find_most_similar_answer(question)
        print(f"Q: {question}\nHR Hive: {answer}\n")

    # Define a loop to continuously prompt the user for input and generate responses
    print("****HR Hive: Simplifying HR, Amplifying Your Workforce****")
    while True:
        # Prompt the user for input
        question = input("\nYou : ")

        # Generate a response to the input text
        response = predict_answer(question, model, tokenizer, answers)

        # Print the response
        print('HR Hive : ',response)

    #!zip -r '/content/fine-tuned-hr-bert.zip' '/content/fine-tuned-hr-bert'



with open('hr_specific_dataset.txt', 'r') as file:
    data = file.readlines()

questions = []
answers = []
for line in data:
    if line.startswith('Q:'):
        questions.append(line.strip().replace('Q: ',''))
    elif line.startswith('A:'):
        answers.append(line.strip().replace('A: ',''))

print('Please wait while the bot is initialized!')

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


stop_words = ["what", "is", "your" "a", "an", "the", "this", "that", "is", "it", "to", "and", "policy"]
stemmer = SnowballStemmer('english')

def preprocess(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

questions_preprocessed = [preprocess(q) for q in questions]


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions_preprocessed)


def get_most_similar_question(question):
    question_preprocessed = preprocess(question)
    question_vector = vectorizer.transform([question_preprocessed])
    similarities = cosine_similarity(X, question_vector)
    most_similar_index = similarities.argmax()
    return most_similar_index
test_questions = ['What is your HR policy on employee benefits and culture?',
                  'How do you ensure a safe and healthy work environment for employees?',
                  'What is your company culture like?',
                  'How do you handle the employee feedback and suggestions in corporation?']
# Step 5: Get the answer
def get_answer(question):
    if any(word in question.lower() for word in ['hello', 'hi', 'welcome', 'hey']):
        tst='\n'.join(test_questions)
        return f"Hello! How can I help you today?\n\nYou can start asking questions like:\n{tst}"
    
    most_similar_index = get_most_similar_question(question)
    answer = answers[most_similar_index]
    return answer







import tkinter as tk

class ChatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HR Hive Bot")

        # Create chat area
        self.chat_area = tk.Text(self.root, height=20, width=90)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.grid(row=0, column=0, padx=10, pady=10)

        # Create input area
        self.input_area = tk.Entry(self.root, width=90)
        self.input_area.grid(row=1, column=0, padx=10, pady=10)

        # Create send button
        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        # Bind enter key to send message
        self.root.bind('<Return>', lambda event: self.send_message())

        # Start GUI loop
        self.root.mainloop()

    def send_message(self):
        # Get input message
        input_message = self.input_area.get()

        # Append message to chat area
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"You: {input_message}\n")
        self.chat_area.insert(tk.END, f"HR Hive: {get_answer(input_message)}\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

        # Clear input area
        self.input_area.delete(0, tk.END)

        # TODO: Send input message to HR Hive chatbot and receive response
        #self.chat_area.insert(tk.END, f"HR Hive: ***********\n")

if __name__ == "__main__":
    ChatGUI()
