import tf_glove

fin = open('text8.txt')
string = fin.read()
corpus = string.split()

model = tf_glove.GloVeModel(embedding_size=128, context_size=1, max_vocab_size=50000)
model.fit_to_corpus(corpus)
model.train(num_epochs=100)
final_embeddings = model.embeddings

fout = open('output_GloVe.txt', 'w')  

for i in range(len(final_embeddings)):
	for j in range(len(final_embeddings[i])):
		if j == 0:
			fout.write(str(corpus[i]) + ' ' + str(final_embeddings[i][j]) + ' ')
		elif j == (len(final_embeddings[i])-1):
			fout.write(str(final_embeddings[i][j]) + '\n')
		else:
			fout.write(str(final_embeddings[i][j]) + ' ')

fin.close()
fout.close()