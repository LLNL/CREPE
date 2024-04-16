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


# CREPE: Uses learnable context vectors to obtain visually grounded text descriptors for union image

{% include add_image.html 
    image="assets/img/pipeline.png"
    caption="" 
    alt_text="Alt text" 
%}


# Quantitative Results

Predicate Estimation Performance: This chart compares the performance of our proposed CREPE method with other state-of-the-art methods on the Visual Genome (VG) dataset, using mean Recall@K (mR@K). The best performing method is highlighted in red, while the second best is in blue. It's worth noting that we are, to our knowledge, the first to report mR@ {5,10,15}, and hence those scores for other methods are not presented.

{% include add_image.html 
    image="assets/img/table.png"
    caption=""

    alt_text="Alt text" 
%}
The R@50 performance of two models CREPE, UVTransE (vision only), while also showing the frequency of each predicate. Predicates are color-coded based on their categories: `Head' (purple), `Mid' (olive), and `Tail' (orange). The recall values are shown as dotted lines, while the predicate frequencies are displayed as blue bars.
{% include add_image.html 
    image="assets/img/long_tail.png"
    caption="" 
    alt_text="Alt text" 
%}

# Qualitative Results on Visual Genome Dataset
Each sub-figure illustrates the relationship between the subject (yellow box) and the object (green box), accompanied by the top five predictions made by CREPE. The accurate prediction is emphasized in red. Notably, in the first column of the third row, although the ground truth label is <flag, on, pole>, CREPE makes a more suitable prediction with <flag, hanging from, pole>, thus indicating that the evaluation metrics can be conservative.

{% include add_gallery.html data="results_VG" %}

# Qualitative Results on Unrel Dataset
Using CREPE to estimate predicates for the Unrel dataset which contains unseen entities and relationship

{% include add_gallery.html data="results_unrel" %}

# Citation

{% include add_citation.html text="@INPROCEEDINGS{ICML_SISTA,
  author={Subramanyam, Rakshith and Jayram, T.S. and Anirudh, Rushil and Thiagarajan, Jayaraman J.},
  booktitle={International Conference on Machine Learning}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2024}}
}


<!-- @INPROCEEDINGS{10096784,
  author={Subramanyam, Rakshith and Thopalli, Kowshik and Berman, Spring and Turaga, Pavan and Thiagarajan, Jayaraman J.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096784}}" %} -->


# Contact
If you have any questions, please feel free to contact us via email: {{ site.contact.emails }}
