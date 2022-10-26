
# Introduction

## Today's Data Trends & Separation of Compute and Storage (The 'Why' Part of Data Orchestration)

Current pace of innovation is slowed by need to reinvent the wheel just for the applications to efficiently access data. A significant effort gets in the way to engineer an application to access data efficiently, rather than focusing on the algorithms and application logic.

When an engineer wants to move an app from on-prem environment to cloud, or a data-scientist who wrote Spark app wants to move to Tensorflow app, etc. Furthermore, as and when there's change to frameworks, storage, or deployment environments, the engineers need to erinvent the wheel allover again to keep data accesses efficient.

As big-data systems evolved from single, all-in-one coupled hardware to co-located compute and storage servers, to a more defined, loosely coupled compute layer, and now to the "Edge Computing", the enterprises began to be challenged on limited access-speeds due to network latencies, between computes like, Spark, Presto, Hive, Tensorflow etc, to storages like, AWS S3, Google Cloud Storage, HDFS, HPE etc.

This page briefly describes the fundamental solution to data-access challenges around the separation of compute and storage the world needs. We need an abstraction between the computation frameworks and storage systems. This is where Data Orchestration comes in.

# What Is Cloud Data Orchestration

Data Orchestration technologies abstract data access across storage systems, virtualize all the data, and present the data via standardized APIs with global namespace to data-driven applications. They also provide a caching functionality to enable fast access to warm data.

Data Orchestration is to data what container orchestration is to containers. Container orchestration is a category of technologies that enables containers to run in any environment agnostic to the hardware that is running the application and ensures that applications are running as intended. Similarly, data orchestration is a category of technologies that enables applications to be compute agnostic, storage agnostic and cloud agnostic. The objective is to make data more accessible to compute no matter where the data is stored.

With a Data Orchestration platform in place, an application developer can work under the assumption that the data will be readily accessible regardless of where the data resides or the characteristics of the storage and focus on writing the application.

# What We Have So Far - Alluxio

Alluxio is an open source Data Orchestration platform. It enables data orchestration in the cloud and fundamentally allows for separation of storage and compute. Alluxio also brings speed and agility to big data and AI workloads and reduces costs by eliminating data duplication and enables users to move to newer storage solutions like object stores.

<img src=https://user-images.githubusercontent.com/105383186/198127485-44308fd4-3d58-4fc9-b9ab-73d7993423c5.png>

Alluxio’s data orchestration in the cloud solves for three critical challenges:
• Data locality: Data is local to compute, giving you memory-speed access for your big data and AI/ML workloads
• Data accessibility: Data is accessible through one unified namespace, regardless of where it resides
• Data on-demand: Data is as elastic as compute so you can abstract and independently scale compute and storage
Alluxio enables data orchestration for compute in any cloud. It unifies data silos on-premise and across any cloud, and reduces the complexities associated with orchestrating data for today’s big data and AI/ML workloads
