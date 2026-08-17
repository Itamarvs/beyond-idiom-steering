## **Final Project proposal \- Steering Idioms Extension**

**Endorsed by Instructor (kai golan hashiloni)**  
Hi, following my emails with Kai, attaching here a short project overview on which we'll extend the [steering idioms paper](https://cdn-uploads.piazza.com/paste/k262tgk1osn1jn/14bcd25454317e4a8d3df400b612d959a8156fe991aa66998fc5f4d843753129/IdioSteer.pdf).  
   
Before the project proposal, i'd mention i'm aware of the late project proposal, Kfir mentioned in the seminar that there’s some flexibility on the proposal deadline, and it was important for me to first properly finish the course material before committing to a final project. I'd appreciate your understanding.

**Project Proposal:**  
**Title:** Drowning in Meaning: Steering Figurative-vs-Literal Interpretation Beyond Idioms  
**Problem description:** The Steering Idioms paper shows that activation steering can shift LLMs between figurative and literal readings of idioms. We ask whether the same steering direction reflects a more general figurative-literal axis that also applies to other figurative expressions (e.g. metaphors/similes), or whether its effect is essentially idiom‑specific.

**Idea & Innovation Highlights**

* Create a small benchmark of non‑idiomatic figurative expressions where both figurative and literal continuations are plausible.

* Apply the idiom‑based steering direction from the Steering Idioms work to this new benchmark and compare its effect to that on idioms.

* Optionally learn a separate steering direction from the new data and compare its behavior and geometry to the idiom‑based one.

* Innovation: Extends steering from idioms to broader figurative language, probing whether the discovered direction captures a general semantic dimension.

**Implementation Steps**

1. Create a compact set of non‑idiomatic figurative expressions (e.g metaphors/similes) and design ambiguous prefixes with both figurative and literal continuations.

2. Use the existing Steering Idioms pipeline to construct an idiom‑based steering direction.

3. Generate steered and unsteered continuations on both IdioSteer and the new figurative benchmark.

4. Compare figurative/literal rates, coherence, and error patterns across expression types.

**Methodology, Datasets, and Models**

* Methodology**:** Follow the activation‑steering setup and evaluation protocol introduced in the Steering Idioms paper (residual‑stream steering with a fixed direction and coefficient grid, plus LLM‑based labeling and a small human‑checked subset).

* Datasets**:** Existing idiom data (for steering direction construction and idiom evaluation) \+ a newly created small benchmark of non‑idiomatic figurative expressions.

* Models**:** 1–2 open‑weights base LLMs from a single family (e.g., LLaMA/Gemma), with a limited configuration grid to allow focused analysis.

If the above doesn't seem aligned with what you had in mind for a project, I also have a simpler backup in mind: testing cross‑lingual idiom steering from English to Hebrew by creating a small Hebrew IdioSteer‑style set and checking whether an English‑trained steering vector transfers in a multilingual model. If needed I’ll formalize it according to the proposal template as well.

Thanks in advance

