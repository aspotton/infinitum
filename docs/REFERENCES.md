# References and Design Influences

Infinitum is not an implementation of any single prior memory framework. Its design combines ideas from long-term agent memory, retrieval-augmented generation, temporal knowledge representation, hierarchical context management, event-sourced systems, and asynchronous/background consolidation.

This document records the papers and systems techniques that most directly influenced the architecture, along with the specific Infinitum design ideas they helped motivate.

> **Note:** These works are design influences, not runtime dependencies. Infinitum's implementation and data model are independent.

## Core research influences

### MemGPT: Towards LLMs as Operating Systems

**Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez. 2023.**  
[arXiv:2310.08560](https://arxiv.org/abs/2310.08560)

MemGPT introduced an operating-system-inspired approach to virtual context management: a model works with a bounded active context while a much larger external memory hierarchy is paged in and out as needed.

**Influence on Infinitum:**

- Treat the model context window as scarce working memory, not as the memory system itself.
- Keep persistent memory outside the model and retrieve only what is useful for the current request.
- Support progressive retrieval so effective memory can be much larger than the model's prompt window.
- Separate memory storage from context compilation.

---

### Generative Agents: Interactive Simulacra of Human Behavior

**Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein. 2023.**  
[arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

Generative Agents maintains a stream of observations, retrieves memories using multiple signals, and periodically synthesizes higher-level reflections from lower-level experiences.

**Influence on Infinitum:**

- Preserve a complete raw event/interaction stream.
- Derive higher-level memory rather than treating raw history as the final memory representation.
- Combine recency, relevance, and importance-like signals during retrieval.
- Build multi-resolution memory layers such as detailed memories and topic summaries.

---

### MemoryBank: Enhancing Large Language Models with Long-Term Memory

**Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, Yanlin Wang. 2024.**  
[AAAI 2024 / DOI: 10.1609/aaai.v38i17.29946](https://doi.org/10.1609/aaai.v38i17.29946)

MemoryBank explores persistent conversational memory with continual updates, memory importance, reinforcement, and time-sensitive retention.

**Influence on Infinitum:**

- Model memory as something that evolves over time rather than append-only retrieval chunks.
- Track reinforcement/observation evidence separately from the raw conversation.
- Include importance and freshness as retrieval signals.
- Plan for richer evidence weighting rather than treating every observation equally.

---

### Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

**Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav. 2025.**  
[arXiv:2504.19413](https://arxiv.org/abs/2504.19413)

Mem0 describes a practical memory-centric architecture that extracts, consolidates, and retrieves salient information from ongoing conversations, including graph-enhanced variants.

**Influence on Infinitum:**

- Learn compact durable memories from interactions rather than retaining every turn in active context.
- Run extraction and consolidation as explicit memory-management stages.
- Keep memory management separate from the main model/provider layer.
- Evaluate memory systems on both quality and operational cost/latency.

---

### A-MEM: Agentic Memory for LLM Agents

**Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang. 2025.**  
[arXiv:2502.12110](https://arxiv.org/abs/2502.12110)

A-MEM uses a Zettelkasten-inspired approach in which memories acquire structured attributes and links, and existing memories can evolve when new related information arrives.

**Influence on Infinitum:**

- Treat memory as structured, evolving knowledge rather than immutable vector-store chunks.
- Maintain topic/type metadata alongside memory content.
- Plan for relationship-aware memory and dynamic linking.
- Allow new information to update the current representation of older knowledge while preserving provenance.

---

### MemRefine: LLM-Guided Compression for Long-Term Agent Memory

**Minjae Kim, Jinheon Baek, Soyeong Jeong, Sung Ju Hwang. 2026.**  
[arXiv:2606.13177](https://arxiv.org/abs/2606.13177)

MemRefine studies storage-budgeted memory compression and argues that surface similarity is useful for proposing consolidation candidates but is not sufficient to determine factual equivalence. It uses an LLM judge to decide whether candidates should be merged, preserved, or removed.

**Influence on Infinitum:**

- Use lexical/embedding similarity to find candidate memories, not as the sole authority for merging them.
- Separate candidate discovery from semantic consolidation decisions.
- Keep deterministic code in control of canonical memory mutation even when an LLM proposes changes.
- Perform bounded periodic consolidation rather than repeatedly sending the entire memory corpus to a model.
- Treat memory budget and value preservation as first-class concerns.

---

### Zep: A Temporal Knowledge Graph Architecture for Agent Memory

**Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef. 2025.**  
[arXiv:2501.13956](https://arxiv.org/abs/2501.13956)

Zep/Graphiti emphasizes temporally aware knowledge representation that can retain changing facts and historical relationships instead of presenting all retrieved facts as timeless truths.

**Influence on Infinitum:**

- Distinguish current derived state from historical evidence.
- Preserve old facts while marking them superseded rather than deleting them.
- Represent contested and changing information explicitly.
- Plan for temporal relationships and time-aware retrieval.

---

### HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models

**Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su. 2024.**  
[arXiv:2405.14831](https://arxiv.org/abs/2405.14831)

HippoRAG combines language models, knowledge graphs, and graph traversal to support associative and multi-hop retrieval over large knowledge collections.

**Influence on Infinitum:**

- Do not assume nearest-neighbor vector search is sufficient for all memory retrieval.
- Plan for relationship-aware and multi-hop expansion on top of lexical and semantic retrieval.
- Keep the retrieval architecture extensible so graph-based signals can later participate in ranking.

---

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela. 2020.**  
[arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

RAG formalized the combination of parametric model knowledge with externally retrieved non-parametric knowledge.

**Influence on Infinitum:**

- Keep durable knowledge outside model weights and retrieve it at inference time.
- Make memory independently updatable without retraining the base model.
- Preserve provenance for retrieved information.
- Extend the retrieval pattern beyond documents to evolving conversational and agent memory.

---

### Reflexion: Language Agents with Verbal Reinforcement Learning

**Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao. 2023.**  
[arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

Reflexion demonstrates that agents can improve future behavior by storing linguistic reflections on prior outcomes rather than modifying model weights.

**Influence on Infinitum:**

- Preserve lessons learned from failures, corrections, and successful recoveries.
- Support episodic and lesson/procedural memory distinct from factual memory.
- Treat memory as a mechanism for learning from experience without fine-tuning the underlying LLM.

---

### Voyager: An Open-Ended Embodied Agent with Large Language Models

**Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar. 2023.**  
[arXiv:2305.16291](https://arxiv.org/abs/2305.16291)

Voyager maintains a growing skill library of reusable behaviors and retrieves relevant skills for future tasks.

**Influence on Infinitum:**

- Distinguish reusable procedures/skills from ordinary factual memories.
- Preserve successful solutions as reusable knowledge.
- Plan for memory types whose value is behavioral reuse rather than factual recall.

---

### LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory

**Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu. ICLR 2025.**  
[arXiv:2410.10813](https://arxiv.org/abs/2410.10813) · [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html)

LongMemEval evaluates long-term chat memory across information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention.

**Influence on Infinitum:**

- Evaluate more than simple factual recall.
- Test temporal updates and whether old information is correctly superseded.
- Measure multi-session reasoning and distractor resistance.
- Test abstention when relevant memory does not exist.
- Separate retrieval quality from final answer quality.

## Systems and architecture techniques

Several important Infinitum design decisions come from established software and information-retrieval patterns rather than a single LLM-memory paper.

### Event sourcing

Infinitum treats the immutable event stream as source-of-truth evidence and treats memories as derived projections.

```text
immutable events
      ↓
derived memories
      ↓
topic/current-state summaries
```

This enables memory consolidation, supersession, and even future rebuilding of the memory database without losing the original observations.

### Materialized views / derived projections

Detailed memories and topic summaries behave like increasingly compact materialized views over the event log. They can be regenerated or replaced without rewriting history.

### CQRS-like separation

Evidence capture, memory mutation, retrieval, and context compilation are intentionally distinct responsibilities. A learning model may propose memory changes, but deterministic application code owns canonical mutation.

### Hierarchical / virtual-memory thinking

Persistent memory is large and durable; the prompt is a bounded working set.

```text
persistent corpus
      ↓
broad retrieval
      ↓
rank / deduplicate / resolve state
      ↓
context compiler
      ↓
small high-value working set
```

The maximum model context window is treated as a ceiling, not a target.

### Hybrid information retrieval

Infinitum combines multiple retrieval signals rather than relying exclusively on embeddings:

- semantic similarity;
- lexical relevance;
- topic relevance;
- freshness;
- importance;
- confidence;
- user/project affinity in the current global-memory implementation;
- later: scope, authority, goal relevance, and graph relationships.

### Provenance tracking

Derived memories retain source-event references. This makes it possible to answer both:

- "What does Infinitum currently believe?"
- "Why does it believe that, and what observations produced the memory?"

### Temporal versioning and supersession

New information can replace current state without deleting the historical state it supersedes.

```text
old memory: active
      ↓ new evidence
old memory: superseded
new memory: active
```

This is fundamental to avoiding the common RAG failure mode where contradictory old and new facts are retrieved together as though both are current.

### Write-behind / asynchronous learning

Memory extraction occurs after the foreground response whenever practical. This keeps learning off the latency-critical user path and allows durable retry through background jobs.

### Priority-aware background scheduling

The intended scheduler separates model work by urgency:

```text
P0  foreground model traffic
P1  per-interaction memory extraction
P2  incremental topic maintenance
P3  periodic deep consolidation
```

Foreground work should take precedence over maintenance work, particularly when the same local inference resource serves both.

### Multi-timescale memory processing

Infinitum deliberately separates memory work into three timescales.

#### Every interaction

Small, immediate, bounded work:

```text
current interaction
+ nearby memories
      ↓
extract durable observations
      ↓
create / reinforce / supersede
```

#### Incremental

Topic-local maintenance after accumulated changes or an idle/debounce period:

```text
existing topic summary
+ changed memories
+ small context sample
      ↓
update topic understanding
```

#### Periodic

More expensive background maintenance:

```text
larger memory corpus
      ↓
cluster into bounded groups
      ↓
consolidate clusters
      ↓
reconcile duplicates / conflicts / stale state
      ↓
rebuild higher-level summaries
```

This keeps memory-management cost closer to the amount of new information/change rather than repeatedly reprocessing everything ever stored.

### Bounded map/reduce-style consolidation

Large memory sets should not be sent wholesale to an LLM. Periodic consolidation should first cluster or partition candidates deterministically, process bounded groups, and then optionally consolidate the compact outputs.

### Value-per-token context compilation

The retrieval layer may find many relevant memories, but the Context Compiler has a separate responsibility: decide which subset is worth spending prompt tokens on.

The long-term goal is to optimize for something closer to:

```text
expected answer value / context token
```

rather than simply top-k similarity.

## Synthesis

The design philosophy can be summarized as:

> **Infinitum treats interaction history as an immutable event stream, memory as an evolving set of provenance-backed derived beliefs, retrieval as a hybrid ranking problem, and the LLM context window as a scarce working-memory cache rather than the memory system itself.**

Or as a pipeline:

```text
raw interactions and tool results
            ↓
      immutable events
            ↓
  extracted observations
            ↓
 create / reinforce / supersede
            ↓
     detailed memories
            ↓
 incremental topic summaries
            ↓
 periodic deep consolidation
            ↓
      hybrid retrieval
            ↓
 state resolution + deduplication
            ↓
      context compiler
            ↓
       bounded prompt
            ↓
            LLM
```

## Evaluation direction

Infinitum's evaluation harness should eventually include at least the following categories, informed especially by LongMemEval and the failure modes exposed by the memory literature:

- simple fact recall;
- paraphrased semantic recall;
- long-gap recall;
- distractor resistance;
- repeated-observation reinforcement;
- duplicate consolidation;
- explicit corrections;
- partial supersession;
- temporal state reconstruction;
- historical versus current-state questions;
- conflicting evidence;
- preference persistence;
- goal persistence;
- procedural/lesson reuse;
- abstention when no relevant memory exists;
- retrieval precision/recall;
- irrelevant-memory injection rate;
- final-answer improvement versus a no-memory baseline;
- added latency and token cost;
- performance as memory corpus size grows.

## Further reading

The references above were selected because they map directly to design decisions already present in Infinitum or explicitly represented in its roadmap. As the project evolves, this file should be updated when new research materially changes the architecture rather than becoming a general bibliography of LLM-memory papers.
