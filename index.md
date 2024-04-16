---
layout: homepage
---

# Summary

We explore the use of Vision-Language Models (VLMs), particularly CLIP, for predicting visual object relationships, simplifying complex graphical models that combine visual and language cues. Our approach, termed CREPE (CLIP Representation Enhanced Predicate Estimation), employs the UVTransE framework to generate translational embeddings for subject, object, and union boxes in scenes using CLIP’s language capabilities. CREPE introduces a novel contrastive training method to refine union-box prompts and significantly improves performance on the Visual Genome benchmark, achieving a 15.3% increase over the previous state-of-the-art with mR@20 scores of 31.95. This demonstrates the potential of CLIP in object relation prediction and suggests further avenues for research in VLM applications.

<br>

<!-- {% include add_video.html 
    youtube_link="https://www.youtube.com/embed/oHNaM_qXQeY" 
%} -->

# CLIP Language Priors are insufficient when naively used
{% include add_image.html 
    image="assets/img/weak_prior.png"
    caption="T-SNE visualization of the predicate representations from UVTransE trained with: (left) CLIP-based image embeddings for subject, object and union box regions; (right) CLIPbased image embedding for the union box, along with CLIP-based text embeddings for subject and object boxes." 
    alt_text="Alt text" 
%}


# Method

{% include add_image.html 
    image="assets/img/pipeline.png"
    caption="CREPE. An illustration of the proposed approach. CREPE uses learnable context vectors along with image-conditioned bias correction to obtain visually grounded text descriptors for an union image. Note that, the CLIP backbone is used to both perform the optimization for text prompt generation as well as producing the (stxt,otxt,uimg) embeddings for UVTransE." 
    alt_text="Alt text" 
%}

<!-- 
<div style="font-size:18px">
  <ol type="a">
  <li><strong>Source training:</strong> Train source classifier and generative model for the source data distribution using StyleGAN-v2.</li>
  <li><strong>Single-shot StyleGAN finetuning:</strong> Fine-tune the source generator using a single-shot example to generate images from the target domain using the SiSTA-U strategy.</li>
  <li><strong>Synthetic data generation:</strong> Generate a synthetic dataset by sampling in the latent space of the target generator for the target domain using SiSTA-G strategy.</li>
  <li><strong>Source-free UDA:</strong> Adapt the source classifier using the synthetically generated target domain data.</li>
</ol>
</div> -->


<!-- {% include add_image.html 
    image="assets/img/website-fig-teaser.png"
    caption="Examples of synthetic data generated using SiSTA. <strong>Please follow the link by clicking the image</strong> to access additional examples for different benchmarks and distribution shifts." 
    alt_text="Alt text" 
    link="https://icml-sista.github.io/"
    height="400"
%} -->



# Results

{% include add_image.html 
    image="assets/img/table.png"
    caption="Predicate Estimation Performance: This chart compares the performance of our proposed CREPE method with other state-of-the-art methods on the Visual Genome (VG) dataset, using mean Recall@K (mR@K). The best performing method is highlighted in red, while the second best is in blue. It's worth noting that we are, to our knowledge, the first to report mR@ {5,10,15}, and hence those scores for other methods are not presented."

    alt_text="Alt text" 
%}

{% include add_image.html 
    image="assets/img/long_tail.png"
    caption="The R@50 performance of two models CREPE, UVTransE (vision only), while also showing the frequency of each predicate. Predicates are color-coded based on their categories: `Head' (purple), `Mid' (olive), and `Tail' (orange). The recall values are shown as dotted lines, while the predicate frequencies are displayed as blue bars." 
    alt_text="Alt text" 
%}

<!-- 
SiSTA significantly improves generalization of face attribute detectors. Here is 1−shot SFDA performance (Accuracy %) averaged across different face attribute detection tasks, under varying levels distribution shift severity (Domains A, B & C) and a suite of image corruptions (Domain D). SiSTA consistently improves upon the SoTA baselines, and when combined with toolbox augmentations matches Full Target DA. -->

{% include add_gallery.html data="results" %}



# Citation

{% include add_citation.html text="@INPROCEEDINGS{ICML_SISTA,
  author={Subramanyam, Rakshith and Jayram, T.S. and Anirudh, Rushil and Thiagarajan, Jayaraman J.},
  booktitle={International Conference on Machine Learning}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2024}}


<!-- @INPROCEEDINGS{10096784,
  author={Subramanyam, Rakshith and Thopalli, Kowshik and Berman, Spring and Turaga, Pavan and Thiagarajan, Jayaraman J.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096784}}" %} -->


# Contact
If you have any questions, please feel free to contact us via email: {{ site.contact.emails }}
