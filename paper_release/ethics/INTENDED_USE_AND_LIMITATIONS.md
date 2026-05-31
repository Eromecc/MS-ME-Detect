# Intended Use and Limitations

MS-ME-Detect is a probabilistic risk-estimation system, not an authorship oracle.

It should not be used as the sole basis for punishment, grading penalties, employment decisions, publication rejection, or other high-stakes adverse action. In education, editorial review, hiring, moderation, or policy settings, model output should trigger human review rather than replace it.

Reports should include false-positive risk. This is especially important for formal human writing, non-native writing, AI-polished text, paraphrased text, mixed human-AI writing, short text, domain-shifted text, and text from genres not represented in the validation data.

Recommended output format:

- risk score
- uncertainty or calibration caveat
- short explanation of the model's evidence families
- explicit limitations and false-positive warning

The system estimates whether a text resembles examples in the training and validation distributions. It does not prove who wrote a text and does not establish intent, misconduct, or originality.
