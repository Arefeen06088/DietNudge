# DietNudge: Postprandial hyperglycemia detection with interpretability

<p align="center">
    <img width="260" alt="Diet Nudge" src="https://github.com/Arefeen06088/DietNudge/assets/50717558/fdea31b4-5b97-493b-adb8-a9d01d5fcde7">
</p>

link to the CGM dataset: https://data.mendeley.com/datasets/c7vx2576y2/1


more information about the dataset can be found at:
1) https://www.sciencedirect.com/science/article/pii/S0010482520302900
2) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8307794/

We are proposing a decision tree based postprandial hyperglycemia (PPHG) detection method which produces linear and interpretable decision boundaries for producing guidelines to follow. The inputs to the model are carb composition, carb and fat amount, insulin amount, time elapsed since last insulin dose, starting blood glucose concentration. Although we are proposing personalized models as they produce better results, we have results for generalized models as well. We also employed recursive feature elimination (RFE) to identify the most significant factors leading to PPHG events.

# BibTeX Citation
If you use information from [DietNudge paper](https://drive.google.com/file/d/1qj4tb76aiTdX5i-WDy73J0co1ElQVaD6/view) in a scientific publication, we would appreciate using the following citations:

    @article{Arefeen2022ForewarningPH,
    title={Forewarning Postprandial Hyperglycemia with Interpretations using Machine Learning},
    author={Asiful Arefeen and Samantha N Fessler and Carol Johnston and Hassan Ghasemzadeh},
    journal={2022 IEEE-EMBS International Conference on Wearable and Implantable Body Sensor Networks (BSN)},
    year={2022},
    pages={1-4}
    }
