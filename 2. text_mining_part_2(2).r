# ============================================================
# TEXT MINING + TOPIC-SENTIMENT CORRELATION (FIXED)
# ============================================================

# --- Load libraries ---
library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud)
library(SnowballC)
library(ggplot2)
library(textdata)
library(topicmodels)
library(widyr)
library(ggraph)
library(igraph)
library(RColorBrewer)
library(plotly)
library(tidyr)
library(ggdendro)

# ------------------------------------------------------------
# Step 1: Load and Clean (Same as before)
# ------------------------------------------------------------
setwd("C:/R/speech") 
files <- list.files(pattern = "*.txt")
speeches <- map_df(files, function(f) {
  tibble(president = tools::file_path_sans_ext(f),
         text = paste(readLines(f, encoding = "UTF-8", warn = FALSE), collapse = " "))
})

data("stop_words")
tidy_speeches <- speeches %>%
  unnest_tokens(word, text) %>%
  mutate(word = str_replace_all(word, "[^a-zA-Z]", "")) %>%
  filter(word != "", nchar(word) > 2) %>%
  anti_join(stop_words, by = "word")

# ------------------------------------------------------------
# Step 2: TOPIC MODELING (LDA)
# ------------------------------------------------------------

# 1. Create Document Term Matrix
dtm <- tidy_speeches %>%
  count(president, word) %>%
  cast_dtm(president, word, n)

# 2. Run Latent Dirichlet Allocation (LDA)
# We use k = 4 because you have 4 speeches loaded
lda_model <- LDA(dtm, k = 4, control = list(seed = 2023))

# 3. Extract Topic-Word probabilities (Beta)
top_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  rename(term = term)

# 4. Extract Document-Topic probabilities (Gamma)
gamma_tidy <- tidy(lda_model, matrix = "gamma")

# 5. Load NRC Sentiment Lexicon
# Note: You may be prompted to download this in the console
nrc <- get_sentiments("nrc")

# 6. Create tidy_sent for the heatmap
tidy_sent <- tidy_speeches %>%
  inner_join(nrc, by = "word")

# ------------------------------------------------------------
# Step 3: Topic-Sentiment Correlation
# ------------------------------------------------------------
topic_sent <- top_terms %>%
  inner_join(nrc, by = c("term" = "word")) %>%
  drop_na(sentiment) %>%                          
  count(topic, sentiment, sort = TRUE)

p_topic_sent <- ggplot(topic_sent,
                       aes(x = sentiment, y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  labs(title = "Figure 5: Sentiment Distribution per Topic",
       x = "Sentiment", y = "Frequency") +
  theme_minimal()

print(ggplotly(p_topic_sent))

gamma_sent <- gamma_tidy %>%
  rename(president = document) %>%
  left_join(tidy_sent %>% count(president, sentiment), by = "president") %>%
  drop_na(sentiment) %>%                          
  group_by(topic, sentiment) %>%
  summarise(avg_gamma    = mean(gamma, na.rm = TRUE),
            total_sent   = sum(n, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(topic_sent_score = avg_gamma * total_sent)

p_heat <- ggplot(gamma_sent,
                 aes(x = factor(topic), y = sentiment, fill = topic_sent_score)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightyellow", high = "darkred") +
  labs(title = "Figure 6: Topic-Sentiment Correlation Heatmap",
       x = "Topic", y = "Sentiment", fill = "Score") +
  theme_minimal()

print(ggplotly(p_heat))

# ------------------------------------------------------------
# Step 4: Bigram Analysis
# ------------------------------------------------------------
speeches_bigrams <- speeches %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

bigrams_separated <- speeches_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "\\d"),
         !str_detect(word2, "\\d"))

bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)

bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

p_bigram <- bigrams_united %>%
  count(bigram, sort = TRUE) %>%
  slice_max(n, n = 20) %>%
  mutate(bigram = reorder(bigram, n)) %>%
  ggplot(aes(n, bigram, fill = n)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Figure 7: Top 20 bigrams (Two-Word Phrases)", x = "Frequency", y = NULL) +
  theme_minimal()

print(ggplotly(p_bigram))

# ------------------------------------------------------------
# Step 5: Network Graph of Co-occurring Words
# ------------------------------------------------------------
bigram_graph <- bigram_counts %>%
  filter(n > 1) %>%  # Lowered from 5 to 1 to accommodate smaller datasets
  graph_from_data_frame()

# Check if the graph has any data before plotting
if (vcount(bigram_graph) > 0) {
  set.seed(2023)
  p_net <- ggraph(bigram_graph, layout = "fr") +
    geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                   arrow = grid::arrow(type = "closed", length = unit(.1, "inches")),
                   end_cap = circle(.07, "inches")) +
    geom_node_point(color = "lightblue", size = 5) +
    geom_node_text(aes(label = name), vjust = 1, hjust = 1, repel = TRUE) + # Added repel
    theme_void() +
    labs(title = "Figure 8: Network of Word Co-occurrences")
  
  print(p_net)
} else {
  message("No bigrams found with the current frequency filter. Try lowering 'n'.")
}