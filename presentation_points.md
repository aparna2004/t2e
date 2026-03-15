# Presentable Points for Viva / PPT / Demo

## Project title
**Event Information Extraction from News Articles using NLP**

## Problem statement
News articles contain a lot of useful event information, but it is buried inside free text.  
This project converts unstructured news text into structured event records.

## Objective
To automatically extract:
- what happened
- who was involved
- where it happened
- when it happened

## Why this project matters
- helps in news summarization
- supports knowledge base creation
- useful for media monitoring
- useful for alert systems and trend tracking

## Input
A raw news article entered by the user.

## Output
Structured event information:
- event type
- trigger word
- participants
- location
- date/time
- confidence score

## Main NLP ideas used
- text preprocessing
- sentence splitting
- named entity extraction
- event trigger detection
- event type mapping
- argument extraction
- structured JSON generation

## Explainability feature
The app provides an optional **step-by-step view** so the user can see:
1. article preprocessing
2. sentence breakdown
3. entity/date/location detection
4. trigger detection
5. final event assembly

## Dataset chosen for training
Recommended:
- **RAMS 1.0**
- **MAVEN**

## Why these datasets were chosen
- designed for event extraction research
- contain annotated event triggers
- suitable for news/document-level text
- useful for building an advanced academic project

## Model idea
The final advanced version can use:
- transformer-based token classification for trigger detection
- context-aware event type classification
- post-processing for arguments, dates, and locations

## Evaluation metrics
- precision
- recall
- F1-score
- event type accuracy
- per-class performance
- confusion analysis

## Strengths of the project
- presentable UI
- explainable output
- structured result format
- easy to demonstrate live
- can be upgraded to a research-level model

## Future improvements
- fine-tune on RAMS/MAVEN
- add better participant role extraction
- add timeline visualization
- support article URL ingestion
- support batch processing of news articles

## One-line conclusion
This project shows how NLP can transform raw news articles into structured event knowledge that is easier to analyze, search, and present.
