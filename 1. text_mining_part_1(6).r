# --- Load libraries (Your original list) ---
library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud) # We are using the original one to avoid the 'markdown' error
library(topicmodels)
library(plotly)

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
# Step 2: Word Frequency per President
# ------------------------------------------------------------
word_freq <- tidy_speeches %>% count(president, word, sort = TRUE)

p_freq <- word_freq %>%
  group_by(president) %>%
  slice_max(n, n = 10) %>%
  ungroup() %>%
  ggplot(aes(x = reorder_within(word, n, president), y = n, fill = president)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ president, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Figure 1: Top 10 Words by President") +
  theme_minimal()

print(ggplotly(p_freq))

# ------------------------------------------------------------
# Step 3: Word Cloud
# ------------------------------------------------------------
word_freq_total <- word_freq %>%
  group_by(word) %>%
  summarise(total = sum(n)) %>%
  arrange(desc(total))

# Open a new window to prevent it from being hidden
if(.Platform$OS.type == "windows") { windows() } else { x11() }

set.seed(123)
# Note: scale is reduced to c(3, 0.3) so words actually fit!
wordcloud(
  words        = word_freq_total$word,
  freq         = word_freq_total$total,
  max.words    = 80,
  random.order = FALSE,
  colors       = brewer.pal(8, "Dark2"),
  scale        = c(3, 0.3) 
)
title("Figure 2: Word Cloud")

# ------------------------------------------------------------
# Step 4: Topic Modeling (LDA)
# ------------------------------------------------------------
dtm <- tidy_speeches %>% count(president, word) %>% cast_dtm(president, word, n)
lda <- LDA(dtm, k = min(5, nrow(as.matrix(dtm))), method = "VEM")

p_topic <- tidy(lda, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(term, beta, topic), beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Figure 3: Top terms per topic (LDA)") +
  theme_minimal()

print(ggplotly(p_topic))

# ------------------------------------------------------------
# Step 5: Sentiment Analysis
# ------------------------------------------------------------
nrc <- get_sentiments("nrc")
p_sent <- tidy_speeches %>%
  inner_join(nrc, by = "word", relationship = "many-to-many") %>%
  count(president, sentiment) %>%
  ggplot(aes(reorder_within(sentiment, n, president), n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ president, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Figure 4: Sentiment distribution per president") +
  theme_minimal()

print(ggplotly(p_sent))