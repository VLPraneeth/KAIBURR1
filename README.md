# Full-Stack Task Management & Command Execution Platform

A cleaned public version of this repository. It brings together the individual task READMEs into a single overview so the whole project can be understood from one place.

## Overview

This repository contains a set of related tasks that together build and deploy a full-stack Task Manager application and a set of machine learning models for consumer complaint classification. Each task lives in its own folder and is summarized below.

## Tasks

### Task 1 - Spring Boot REST API (Backend)

A Java Spring Boot backend that exposes a REST API for managing tasks (records that represent shell commands to be run). Task data is stored in MongoDB. Core classes include the Task model, the TaskExecution model, and a TaskController that provides the create, search, and retrieve endpoints consumed by the frontend.

### Task 2 - Containerization and Kubernetes Deployment

Deployment configuration for the Task Manager application. It includes a Dockerfile that packages the Spring Boot app on an OpenJDK 17 base image (exposing port 8080), plus Kubernetes manifests: a Deployment (running replicas of the app with resource limits and a MongoDB connection via environment variables) and a NodePort Service that exposes the app to the cluster.

### Task 3 - React Frontend

A React user interface for the Spring Boot backend. It lets users create, search, and manage tasks that represent shell commands to be executed, communicating with the Spring Boot REST API via POST and GET requests.

Run it with:

```bash
npm install   # install dependencies
npm start     # start the dev server
```

Features include a task management dashboard, a form to create new tasks (Task ID, Task Name, Owner Name, and Shell Command), a search component to find tasks by name, and backend/MongoDB data verification.

### Task 5 - Consumer Complaint Classification Model

A machine learning model that classifies consumer complaints into predefined categories from the complaint text. Written in Python, it uses TF-IDF vectorization for feature extraction and an XGBoost classifier (n_estimators=100, max_depth=6, learning_rate=0.1).

The workflow covers data loading and filtering, preprocessing (removing nulls, lowercasing, punctuation removal, stopword filtering, and mapping categories to labels 0-3), feature extraction, model training, and evaluation via accuracy, precision, recall, F1-score, a confusion matrix, and an error distribution plot. It reports an overall accuracy of about 95.7%.

Run it with:

```bash
python model.py
```

### Task 6 - Text Classification (Additional Models)

A Python text-classification script that experiments with additional algorithms for complaint categorization, including Multinomial Naive Bayes, Logistic Regression, and SVM. It uses pandas, scikit-learn, and NLTK for preprocessing (tokenization, stopword removal, stemming) and TF-IDF vectorization, and evaluates results with a classification report, confusion matrix, and accuracy score.

## Notes

Personal names are intentionally omitted. This README consolidates the per-task README files; some task folders did not contain written descriptions, so those summaries are derived from the code in that folder.
