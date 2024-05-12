===
FAQ
===


    * **Is DREAMS_MC model agnostic?**

        Yes. By design, it can generate model card for any model type. 


    * **At what stage in the model training pipeline should the DREAMS_MC function be applied?**
        
        The dreams_mc function should be utilized after the model training and validation steps to generate the model card. If model card is intended for test set, call the function after  the testing step.

    
    * **Are there examples showcasing a complete pipeline using DREAMS_MC?**

        Yes, we provide notebook with examples of complete pipeline using DREAMS_MC. The notebooks are available `here <https://github.com/LucidJun/DREAM/tree/main/notebooks>`_. If you want to share a notebook with additional pipelines, please feel free to reach us.

    * **Why should I use DREAMS_MC?**

        DREAMS assists researchers to practice a responsible, understandable, and transparent work, aligning with broader goals of ethical AI development. DREAMS_MC allows you to produce the model card which is 
        useful for AI practitioners as they provide essential documentation that enhances transparency, facilitates user understanding, and supports responsible deployment by detailing a model’s performance, biases, and operational parameters.

    * **How can I give the authors credit for their work?**
                                                                                                                                                                                                    
        If you use DREAMS_MC we would appreciate if you star our `GitHub <https://github.com/LucidJun/DREAM>`_ repository. In case you use DREAMS_MC during your research, we would be happy if you can `cite our work <site>`_.