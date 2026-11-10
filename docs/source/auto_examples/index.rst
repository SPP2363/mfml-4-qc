:orphan:

Examples
=========

The below are some examples on how to get started with MFML for your own QC workflow. 


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial covers the basics of data handling, visualization, and structural representation generation in MFML-QC. We will use an inbuilt dataset for the benzene molecule.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_01_load_and_represent_thumb.png
    :alt:

  :doc:`/auto_examples/01_load_and_represent`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A Multi-Fidelity Dataset: MD Benzene</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will build our first kernel based machine learning model with the in-built Kernel Ridge Regressor module. We will focus on a single fidelity KRR model that predicts the excitation energies of the benzene molecule from the dataset provided in this package.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_02_krr_learning_curve_thumb.png
    :alt:

  :doc:`/auto_examples/02_krr_learning_curve`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kernel Ridge Regression</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use the MFML-QC package to load the built-in Benzene trajectory dataset, manually extract a multi-fidelity subset using a top-down approach (that is start with the highest fidelity then move down the fidelity hierarchy), and train an MFML model to predict high-fidelity excitation energies.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_03_mfml_basics_thumb.png
    :alt:

  :doc:`/auto_examples/03_mfml_basics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Basics of MFML</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the &quot;duck typing&quot; flexibility of the ModelMFML orchestrator. DUck typing is summarized well with &#x27;If it walks like a duck, quacks like a duck, and swims like a duck, it must be a duck.&#x27; Here, we use it to mean that we can use any ML architecture that has certain attributes. While MFML-QC provides an ultra-fast, built-in Kernel Ridge Regressor (KRR), you are not forced into using it.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_04_duck_typing_mfml_thumb.png
    :alt:

  :doc:`/auto_examples/04_duck_typing_mfml`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Duck Typing for MFML</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="As we saw in on of the previous examples, learning curves are a vital metric for evaluating kernel-based machine learning methods. They depict the relationship between the model&#x27;s prediction error and the amount of training data provided.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_05_mfml_learning_curve_thumb.png
    :alt:

  :doc:`/auto_examples/05_mfml_learning_curve`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MFML Cross-Validation and Learning Curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates the use of optimized multifidelity machine learning (o-MFML) with a comparison of learning curves for:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_06_omfml_thumb.png
    :alt:

  :doc:`/auto_examples/06_omfml`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optimized MFML</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use the OrcaEngine to perform quantum chemistry calculations.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_07_orca_oracle_thumb.png
    :alt:

  :doc:`/auto_examples/07_orca_oracle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Interfacing with QC Oracles (ORCA)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In machine learning for quantum chemistry, molecular geometries are often abundant but labeling them with the properties (e.g., running expensive quantum chemistry calculations for excitation energies) is costly. Active Learning (AL) solves this by iteratively selecting only the most informative geometries from an unlabeled pool to add to the training set.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_08_pool_based_active_learning_thumb.png
    :alt:

  :doc:`/auto_examples/08_pool_based_active_learning`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Active Learning Protocols</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to predict molecular forces as the negative gradient of MFML-predicted energies, computed via automatic differentiation (backpropagation through the Coulomb matrix descriptor and kernel evaluation).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_09_autodiff_forces_thumb.png
    :alt:

  :doc:`/auto_examples/09_autodiff_forces`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Autodifferentiable Force Prediction with MFML</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates vector-valued KRR for predicting electronic dipole moments directly as 3D vectors. We compare the use of an isotropic coupling kernel, which essentially boils down to using individual KRR for each component of the dipole moment and the use a output covariance matrix that is learned during training. Readers will note that this latter approach resembles the multi-task GPR.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_10_vector_krr_dipole_thumb.png
    :alt:

  :doc:`/auto_examples/10_vector_krr_dipole`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Vector KRR for Dipole Moments</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In J. Martinka, M. Pederzoli, M. Barbatti, P. O. Dral, J. Pittner, A simple approach to rotationally invariant machine learning of a vector quantity, J. Chem. Phys. 161, 174104 (2024). DOI: 10.1063/5.0230176 the authors present a simple pre and post processing heuristic to train a KRR model for prediction of challenging vector quantities. This method is called the RPR. RPR (Rotate-Predict-Rotate) transforms each geometry into a canonical inertia-tensor frame before computing descriptors and rotates the predictions back to the lab frame, ensuring equivariant predictions.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_11_rpr_vector_krr_thumb.png
    :alt:

  :doc:`/auto_examples/11_rpr_vector_krr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">RPR Frame Canonicalization for Vector KRR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We will compares the vector MFML against 3 independent scalar MFML models (one per dipole component) to check if the vector MFML does add any improvement at all.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_12_vector_mfml_dipole_thumb.png
    :alt:

  :doc:`/auto_examples/12_vector_mfml_dipole`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Vector MFML for Dipole Moments</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example we will demonstrate a full set of learning curves for the following cases of single fidelity and multifidelity ML models. We will be using the KRR basic architecture.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_13_vector_mfml_learning_curves_thumb.png
    :alt:

  :doc:`/auto_examples/13_vector_mfml_learning_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Learning Curve generation for: RPR × T × MFML</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/01_load_and_represent
   /auto_examples/02_krr_learning_curve
   /auto_examples/03_mfml_basics
   /auto_examples/04_duck_typing_mfml
   /auto_examples/05_mfml_learning_curve
   /auto_examples/06_omfml
   /auto_examples/07_orca_oracle
   /auto_examples/08_pool_based_active_learning
   /auto_examples/09_autodiff_forces
   /auto_examples/10_vector_krr_dipole
   /auto_examples/11_rpr_vector_krr
   /auto_examples/12_vector_mfml_dipole
   /auto_examples/13_vector_mfml_learning_curves


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
