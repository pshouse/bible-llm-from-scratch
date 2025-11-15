# Complete LLM Tutorial Curriculum

## Current Status: Lessons 1-5 Complete ✅

### ✅ Available Now (Fully Implemented)

**Part 1: Understanding the Data**
- ✅ Lesson 1: Exploring the Bible Corpus (20 min)
- ✅ Lesson 2: Tokenization - Converting Text to Numbers (25 min)

**Part 2: Building Blocks**
- ✅ Lesson 3: Embeddings - Representing Words as Vectors (25 min)
- ✅ Lesson 4: Attention Mechanism - How Models Focus (30 min)
- ✅ Lesson 5: Multi-Head Attention - Parallel Understanding (25 min)

### 📋 Coming Soon (Planned Curriculum)

**Part 3: The Transformer Architecture**
- Lesson 6: Feed-Forward Networks (20 min)
  - Why attention alone isn't enough
  - Position-wise feed-forward networks
  - GELU activation function
  - Implementing FFN layers

- Lesson 7: Layer Normalization (15 min)
  - Training stability problems
  - Batch norm vs layer norm
  - Pre-norm vs post-norm
  - Residual connections

- Lesson 8: Complete Transformer Block (25 min)
  - Combining all components
  - Stacking multiple layers
  - Full GPT-style architecture
  - Testing the complete model

**Part 4: Training the Model**
- Lesson 9: Loss Functions (20 min)
  - Cross-entropy loss explained
  - Why it works for language modeling
  - Calculating loss on our Bible corpus
  - Perplexity metric

- Lesson 10: Training Loop (30 min)
  - Forward pass through the model
  - Backward pass and gradients
  - Parameter updates
  - Tracking training metrics

- Lesson 11: Optimization (25 min)
  - SGD vs Adam vs AdamW
  - Learning rate schedules
  - Gradient clipping
  - Mixed precision training

**Part 5: Text Generation**
- Lesson 12: Sampling Strategies (25 min)
  - Greedy sampling
  - Temperature scaling
  - Top-k and top-p sampling
  - Generating Bible-style text

- Lesson 13: Evaluation (20 min)
  - Perplexity calculation
  - Human evaluation
  - Prompt engineering
  - Model limitations and next steps

## Total Time Estimate

- **Available Now**: ~2 hours (Lessons 1-5)
- **Full Curriculum**: ~5 hours (All 13 lessons)

## How to Continue Learning

While lessons 6-13 are being developed, you can:

1. **Review and experiment** with lessons 1-5
   - Modify parameters
   - Try different configurations
   - Implement variations

2. **Study the existing code**:
   - `bible_llm.py` - See the full architecture
   - `the_verdict.py` - Original learning examples

3. **Read the documentation**:
   - `README.md` - Project overview
   - `CLAUDE.md` - Technical details

4. **Practice implementations**:
   - Build your own tokenizer
   - Implement attention from scratch
   - Create custom embedding schemes

## Request Additional Lessons

Want lessons 6-13 created? Just ask! Each lesson will be:
- Interactive with examples
- Hands-on with code
- Include experiments
- End with a quiz

The foundation you've learned (lessons 1-5) covers 60% of understanding how LLMs work!
