# CS412 - RESEARCH MINI-PROJECT

## Bisaya Post-Class Speech and Affective State Recognition Among Computer Science Students

### I. Project Overview

This research mini-project investigates the feasibility of recognizing the **post-class affective** **states of Computer Science students from short, naturally spoken Bisaya responses**. Unlike acted Speech Emotion Recognition (SER) datasets, participants in this project will not be instructed to portray a particular emotion. Instead, students will provide a short and natural spoken response immediately after attending a class and independently report how they actually feel. The collected speech recordings, self-reported affective states, and educational context will be used to create a class-level **Bisaya Post-Class Speech and Affect Dataset** and to develop a preliminary machine learning-based Speech Emotion Recognition system. The project follows the research pipeline: **Research Protocol → Informed Consent → Natural Speech Collection → Self-Reported** **Affect → Audio Preprocessing → Acoustic Feature Extraction → Machine Learning →** **Evaluation → Interpretation**

# II. Research Objectives

The mini-project aims to:

1. determine the affective states reported by Computer Science students immediately after attending their classes;
2. examine the acoustic and prosodic characteristics of short Bisaya post-class speech responses;
3. investigate whether acoustic and prosodic speech features can be used to classify students' self-reported affective states;
4. compare the performance of appropriate machine learning models for Bisaya affect recognition; and
5. explore possible relationships between students' post-class affect and educational context, including class activity, year level, and class session.

# III. Research Questions

The class will collectively investigate the following questions: **RQ1.** What affective states do Computer Science students report immediately after their classes? **RQ2.** How do post-class affective states vary according to year level, class activity, and class session? **RQ3.** What acoustic and prosodic characteristics are exhibited in short Bisaya speech responses across different self-reported affective states? **RQ4.** To what extent can machine learning models classify students' self-reported affective states using acoustic and prosodic features extracted from their Bisaya speech responses? **RQ5.** Which machine learning model provides the best performance for the collected Bisaya post-class speech dataset?

# IV. Participants and Group Assignments

Each group will collect data from its assigned year level.

|Group|Assigned Participants|Minimum Valid Responses|
|---|---|---|
||Group 1 First-Year CS Students Group 2 First-Year CS Students Group 3 Second-Year CS Students Group 4 Second-Year CS Students Group 5 Third-Year CS Students Group 6 Third-Year CS Students Group 7 Fourth-Year CS Students|40 40 40 40 40 40 40|

The class should obtain a minimum of **280 valid participant responses**. Groups assigned to the same year level must coordinate to minimize duplicate participants. Each participant should ordinarily contribute **one post-class observation** to the primary dataset unless the approved research design specifically allows repeated observations.

# V. Participant Eligibility

A participant may be included when the student:

- is currently enrolled in the assigned CS year level;
- has just completed a scheduled class;
- voluntarily agrees to participate;
- agrees to the recording of their short speech response; and
- completes the required post-class self-report.
Do not collect responses from students who do not provide appropriate consent.

# VI. Research Ethics and Informed Consent

This activity involves human participants and the recording of their voices. Before collecting any data, researchers must explain:

- the purpose of the study;
- what the participant will be asked to do;
- what information will be collected;
- that their voice will be recorded;
- that participation is voluntary;
- that they may refuse or discontinue participation;
- that declining will have no academic penalty;
- how their data will be stored and protected; and
- that de-identified/pseudonymized data and research findings may be used for academic research, presentation, and publication, subject to the approved research protocol.
### Important

### No recording may begin until informed consent has been obtained.

Do not use the participant's name or student ID as the research identifier. Assign an anonymous participant code such as: Y1-001 Y2-017 Y3-026

The identity of individual participants must not appear in the machine learning dataset or research report. Because the class intends to potentially publish the research findings, data collection must begin only after compliance with the applicable institutional research ethics and data privacy requirements.

# VII. Standard Data Collection Protocol

All seven groups must follow the same procedure.

## Step 1 — Approach the Participant

Approach an eligible CS student immediately after the student's class. Explain the study and obtain informed consent. Do not pressure students to participate.

## Step 2 — Record the Educational Context

Ask the participant what activity primarily occurred during the class that just ended. Use one of the following categories:

- Lecture/Discussion
- Laboratory Activity
- Programming/Coding Activity
- Quiz
- Examination
- Presentation/Reporting
- Group Activity
- Recitation
- Project/Practical Activity
- Other
Also record:

- Year Level
- Class Activity

- Class Session
Class Session should be classified consistently as: **Morning** or **Afternoon–Evening** The exact time boundary used to distinguish the two sessions must be defined before data collection and applied consistently by all groups.

# VIII. Natural Bisaya Speech Response

After recording the contextual information, ask: **“Human sa imong klase, pilia ang usa ka pulong nga pinakaduol sa imong reaksyon karon.** **Isulti lang kini sa natural nga paagi.”**

The participant may select **one** of the following:

- **Salamat**
- **Ambot**
- **Grabe**
- **Lisod**
- **Sayon**
- **Lingaw**
- **Kapoy**
Any response is acceptable. There is no correct or preferred word. **Important Recording Rule** The participant must say the selected word **naturally**. Researchers must NOT say: “Say it happily.” “Say it angrily.”

“Say it sadly.” “Sound tired.” or provide any other instruction about emotional expression. The purpose is to capture **naturally expressed affect**, not acted emotion.

# IX. Self-Reported Affective State

Immediately after the audio recording, obtain an independent measure of how the participant actually feels. Ask: **“How do you feel right now after your class?”** Record the participant's response. For quantitative analysis, the participant should also provide the following two ratings.

**A. Valence** **How pleasant or unpleasant do you feel right now?** 1—Very Unpleasant 2 — Unpleasant 3 — Neutral 4 — Pleasant 5—Very Pleasant
### B. Arousal

### How is your current level of energy or activation?

1—Very Low 2 — Low 3 — Moderate 4 — High 5—Very High The participant's **self-report**, rather than the researchers' interpretation of the voice, will serve as the reference for the participant's affective state.

# X. Audio Recording Protocol

As much as possible, recordings should be collected under similar conditions. Researchers should:

- use the same or comparable recording device/settings;
- record in a reasonably quiet location;
- maintain an appropriate microphone distance;
- avoid unnecessary background conversations;
- record only the intended spoken response;
- avoid audio filters or voice enhancement effects;
- avoid modifying pitch, speed, or other speech characteristics; and
- verify that the response is clearly audible.
Do not ask participants to repeatedly perform the word until it sounds sufficiently emotional. Natural expression is the priority.

# XI. Data to be Collected

Each participant record should contain:

|Variable|Description|
|---|---|
|Participant Code|Anonymous research identifier|
|Year Level|1st, 2nd, 3rd, or 4th Year|
|Class Activity|Activity immediately before collection|
|Session|Morning / Afternoon–Evening|
|Spoken Word|Selected Bisaya response Self-Reported Feeling Participant's own description|
|Valence|1–5|
|Arousal|1–5|
|Audio Filename|Corresponding WAV recording|

Optional non-identifying variables may be included only when they have a clear research purpose and are covered by the approved research protocol.

# XII. File Naming Convention

Use: ParticipantCode_YearLevel_Session_Word.wav Example: Y2-017_Y2_PM_Grabe.wav Never include the participant's name or student number in the filename.

# XIII. Exploratory Research Analysis

Before developing the SER model, describe the collected dataset. Investigate patterns involving: **Year Level × Affect** **Class Activity × Affect** **Class Session × Affect** **Spoken Word × Affect** Examples of questions that may be explored include:

- Do morning and afternoon-evening classes exhibit different post-class affective patterns?
- Are certain class activities associated with higher or lower valence?
- Do reported energy levels differ across year levels?
- Which Bisaya words are most frequently selected after particular class activities?
- Can the same Bisaya word represent different affective states depending on how it is spoken?
These analyses form the **educational affect component** of the research.

# XIV. Speech Preprocessing and Feature Extraction

Apply a consistent preprocessing procedure to all valid audio recordings. Extract appropriate acoustic and prosodic characteristics based on the techniques covered in the previous SER laboratory exercises. Possible features include:

- Fundamental Frequency/Pitch
- RMS Energy
- Zero-Crossing Rate
- MFCCs
- Delta MFCCs
- Delta-Delta MFCCs
- Temporal/rhythm characteristics
- Other justified acoustic features
Each audio recording should ultimately correspond to one structured observation in the machine learning feature dataset.

# XV. Machine Learning Experiment

Develop a machine learning model that investigates whether the acoustic characteristics of the recorded speech can predict the participant's self-reported affect. **Primary Input** **Acoustic and prosodic speech features** **Reference Outcome** **Self-reported affect** The spoken word, year level, class activity, and class session should initially be treated as **contextual variables rather than acoustic predictors**. This is important because the objective is to determine whether the **speech signal itself** **contains affective information**, rather than allowing the model to predict affect simply from words such as *Kapoy*, *Lingaw*, or *Lisod*. Students may compare appropriate machine learning algorithms learned in the course.

# XVI. Speaker-Independent Evaluation

The evaluation must minimize the possibility that the model simply learns the characteristics of individual speakers. When a participant contributes multiple recordings, recordings belonging to the same participant must not be divided between the training and testing datasets. Where appropriate, use a **speaker-independent or participant-grouped validation** **strategy**. The research question should therefore be interpreted as: **Can the model recognize the affective state of a student whose speech was not used to train the model?**

Evaluate performance using appropriate classification metrics and confusion matrices.

# XVII. Lexically Controlled Analysis

An additional analysis should investigate recordings involving the **same spoken word**. For example: **“Grabe” + positive self-report** versus **“Grabe” + negative self-report** This allows the researchers to investigate whether acoustic and prosodic characteristics distinguish affect even when the lexical content is held constant. This analysis is particularly important because some of the selected Bisaya words inherently carry semantic information related to the participant's experience.

# XVIII. Expected Research Outputs

Each group must submit:

1. Minimum of 40 valid participant observations
2. Required consent documentation
3. Properly coded research metadata
4. Properly named audio recordings
5. Data quality report
6. Extracted acoustic feature dataset
7. Exploratory analysis
8. Machine learning experiment
9. Model evaluation
10.Interpretation of findings
11.Research limitations
12.Group research report
The complete class dataset may subsequently be used for a consolidated research paper, subject to the approved research protocol and applicable institutional requirements.

# XX. Expected Class Research Output

At the completion of the project, the class should produce: **Dataset** **Bisaya Post-Class Speech and Affect Dataset** **Computational Output** **Prototype Bisaya Post-Class Affective State Recognition System** **Research Output** A pilot study investigating the feasibility of recognizing naturally expressed post-class affect from short Bisaya speech responses among Computer Science students.

# Important Research Principle

The selected Bisaya word is **not automatically the participant's emotion**. For example: **“Kapoy” ≠ automatically sad**

### “Grabe” ≠ automatically angry

### “Lingaw” ≠ automatically happy

The participant's independently reported affect provides the reference information. The research therefore investigates whether characteristics of **how the word is spoken** contain information related to the participant's affective state.