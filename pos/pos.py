'''
Created on Aug 22, 2013

@author: tvandrun and Stirling Joyner
'''


import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
from nltk import word_tokenize
from math import log 


# 1. Load a (training) corpus.

# In the code below, the corpus will be
# referred to by variable all_text
raw = open("corpus/lovecraft_training.txt").read()
all_text = nltk.Text(word_tokenize(raw))

# make the training text lowercase
all_text_lower = [x.lower() for x in all_text]
freq_dist = FreqDist(all_text_lower)

# make a reduced vocabulary
# In choosing a vocabulary size there is a trade-off.
# A larger vocabulary will in principle make for a more accurate
# tagger, but will be slower and will have a greater risk of underflow.
vocab_size = 500
vocab = sorted(freq_dist.keys(), key=lambda x : freq_dist[x], reverse=True)[:vocab_size]
# "***" is for all out-of-vocabulary types
vocab = vocab + ["***"]




# 2. Make a reduced form of the PennTB tagset
penntb_to_reduced = {}
# noun-like
for x in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'EX', 'WP'] :
	penntb_to_reduced[x] = 'N'
# verb-like
for x in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'TO'] :
	penntb_to_reduced[x] = 'V'
# adjective-like
for x in ['POS', 'PRP$', 'WP$', 'JJ', 'JJR', 'JJS', 'DT', 'CD', 'PDT', 'WDT', 'LS']:
	penntb_to_reduced[x] = 'AJ'
# adverb-like
for x in ['RB', 'RBR', 'RBS', 'WRB', 'RP', 'IN', 'CC']:
	penntb_to_reduced[x] = 'AV'
# interjections
for x in ['FW', 'UH'] :
	penntb_to_reduced[x] = 'I'
# symbols
for x in ['SYM', '$', '#'] :
	penntb_to_reduced[x] = 'S'
# groupings
for x in ['\'\'', '(', ')', ',', ':', '``'] :
	penntb_to_reduced[x] = 'G'
# end-of-sentence symbols
penntb_to_reduced['.'] = 'E'

reduced_tags = ['N', 'V', 'AJ', 'AV', 'I', 'S', 'G', 'E']

# 3. tag the corpus
all_tagged = nltk.pos_tag(all_text)
all_reduced_tagged = []
# 4. make the probability matrices

# a tally from types to tags
word_tag_tally = {y:{x:0 for x in reduced_tags} for y in vocab}
# a tally from tags to next tags
tag_transition_tally = {y:{x:0 for x in reduced_tags} for y in reduced_tags}
# A tally of each tag
tag_tally = {t:0 for t in reduced_tags}

previous_tag = 'E' # "ending" will be the dummy initial tag
for (word, tag) in all_tagged :
	word = word.lower()
	# What if the word is not in the vocabulary?
	if word not in vocab:
		word = "***"
	# Convert the tag to the reduced tag
	tag = penntb_to_reduced[tag]
	# Update the tag tally
	tag_tally[tag] = tag_tally[tag] + 1
	# Update the word tag tally
	word_tag_tally[word][tag] = word_tag_tally[word][tag] + 1
	# Update the tag transition tally
	tag_transition_tally[previous_tag][tag] = tag_transition_tally[previous_tag][tag] + 1
	previous_tag = tag
# now, make the actual transition probability matrices 
trans_probs = {}
for tg1 in reduced_tags :
	# if tg1 never occurs
	if tag_tally[tg1] == 0:
		continue
	# For each tag tg1 compute the probabilities for transitioning to
	# each tag, using relative frequency estimation.
	for tg2 in reduced_tags:
		# That would mean dividing the number of times tg2 follows tg1 by
		# the absolute number of times t1 occurs.
		trans_probs[(tg1,tg2)] = tag_transition_tally[tg1][tg2]/tag_tally[tg1]
	 
# similarly for the emission (observation) probabilities
emit_probs = {}
# For each tag compute the probabilities for emitting each word.
for tag in reduced_tags :
	for word in vocab:
		#number of times this tag emitted this word/number of times this tag appeared
		emit_probs[(tag,word)] = word_tag_tally[word][tag]/tag_tally[tag]

# 5. implement Viterbi. 
# Write a function that takes a sequence of tokens,
# a matrix of transition probs, a matrix of emit probs,
# a vocabulary, a set of tags, and the starting tag

def pos_tagging(sequence, trans_probs, emit_probs, vocab, tags, start) :
	a = trans_probs
	b = emit_probs
	states = tags
	viterbi = [[] for o in sequence]
	for t in range(len(sequence)):
		for s in range(len(states)):
			# Initialization
			if t == 0:
				try:
					viterbi[t].append((a[(start,states[s])] * b[(states[s],sequence[0])] , -1)) # (prob, backpointer)
				except ValueError:
					viterbi[t].append((0,-1))
			else:
				# Find greatest probability of state in previous column transitioning to this state
				choices = []
				for ss in range(len(states)):
					try:
						choices.append((viterbi[t-1][ss][0] * a[(states[ss],states[s])] * b[(states[s],sequence[t])] , ss))
					except ValueError:
						choices.append((0.0,ss))
				choice = max(choices, key=lambda x: x[0])
				viterbi[t].append((choice))
	# Go back through the trelis and find the most probable path
	tagged = []
	for i in range(len(viterbi)-1 ,-1,-1):
		# For the last column (first visited)
		if i == len(viterbi)-1:
			# Find greatest probability of the last column 
			choices = []
			for s in range(len(states)):
				choices.append((viterbi[i][s] , s, viterbi[i][s][1])) # (prob, state, backpointer)
			choice = max(choices, key=lambda x: x[0])
			backpointer = choice[2]
			tagged.append((sequence[i],states[choice[1]]))
		else:
			# Follow the backpointer to the previous tag
			tagged.append((sequence[i] , states[backpointer]))
			# set the new backpointer
			backpointer = viterbi[i][backpointer][1]
	tagged.reverse()
	return tagged

# 6. try it out: run the algorithm on the test data
words = word_tokenize(open("corpus/test.txt").read())
words_processed = []
# Strip out the non-vocab words
for word in words:
	if word.lower() not in vocab:
		word = "***"
	words_processed.append(word.lower())
test_sample = nltk.Text([word for word in words_processed])
print(test_sample)
test_tagged = pos_tagging(test_sample, trans_probs, emit_probs, vocab, reduced_tags, 'E')            
print(test_tagged)



