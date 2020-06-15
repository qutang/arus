
# Overview

ARUS provides a flexible computational framework to manage the data processing flow. It provides a unified interface for different data sources, including sensory data from files and real-time devices, as well as annotation or event-based data from files and devices (annotation tools). It uses a computational graph to support building flexible data processing schemes and provides a set of common used data processing operators. Finally, it is built on multi-process or multi-thread processing to utilize the multi-core CPU architecture of modern computers. 

__Highlights__

1. Unified interfaces (`Stream`, `Generator`) for different data sources.
2. Flexible computational graph (`Node`) and common data processing operators (`Segmentor`, `Synchronizer`, `Scheduler`, `Processor`).
3. Built-in support for multi-thread and multi-core processing.

## Package architecture

[![](https://docs.google.com/drawings/d/e/2PACX-1vQ4ukbtsZiIw4ZNY6rNQj8_MCUxJ3Gr4U5xisx0aecciZhttOUTE4zKHyv1tiUOQkPD2BMxJr4YZtNS/pub?w=951&amp;h=432)](https://docs.google.com/drawings/d/1-vPmqKjMxtBIcGQVgiZcJBeqZEKiuGGm_3aRSoM2OLA/edit?usp=sharing)

__Figure 1. ARUS package architecture__

## Data flow module

Data flow module provides building blocks (classes) to support flexible computational graph, unified data interface, and common data flow operators.

The following class UML demonstrates the data flow building blocks (classes) and their relationships.

??? info "Class diagram: Core building blocks"
    [![Core building blocks](https://mermaid.ink/img/eyJjb2RlIjoiY2xhc3NEaWFncmFtXG4gICAgTm9kZSBvLS0gT3BlcmF0b3JcbiAgICBPcGVyYXRvciA8fC0tIFNlZ21lbnRvclxuXHRPcGVyYXRvciA8fC0tIEdlbmVyYXRvclxuICAgIE9wZXJhdG9yIDx8LS0gU3luY2hyb25pemVyXG4gICAgT3BlcmF0b3IgPHwtLSBQcm9jZXNzb3JcbiAgICBPcGVyYXRvciA8fC0tIFBpcGVsaW5lXG4gICAgU3RyZWFtIG8tLSBTZWdtZW50b3JcbiAgICBTdHJlYW0gby0tIEdlbmVyYXRvclxuICAgIFBpcGVsaW5lIG8tLSBTdHJlYW1cbiAgICBQaXBlbGluZSBvLS0gU3luY2hyb25pemVyXG4gICAgUGlwZWxpbmUgby0tIFByb2Nlc3NvclxuICAgIFByb2Nlc3NvciBvLS0gU2NoZWR1bGVyXG4gICAgY2xhc3MgTm9kZXtcbiAgICAgICAgK2dldF9uYW1lKCkgc3RyXG4gICAgICAgICtnZXRfdHlwZSgpIE8uVHlwZVxuICAgICAgICArZ2V0X29wKCkgT3BlcmF0b3JcbiAgICAgICAgK2dldF9zdGF0dXMoKSBPLlN0YXR1c1xuICAgICAgICArc3RhcnQoKVxuICAgICAgICArc3RvcCgpXG4gICAgICAgICtjb25zdW1lKHBhY2spXG4gICAgICAgICtwcm9kdWNlKClcbiAgICB9XG4gICAgY2xhc3MgT3BlcmF0b3J7XG4gICAgICAgICtydW4oKSpcbiAgICAgICAgK3NldF9jb250ZXh0KClcbiAgICAgICAgK3N0b3AoKVxuICAgICAgICArZ2V0X3Jlc3VsdCgpIHBkLkRhdGFmcmFtZSwgZGljdFxuICAgIH1cbiAgICBjbGFzcyBQaXBlbGluZXtcbiAgICAgICAgK3N0YXJ0KHN0YXJ0X3RpbWUpXG4gICAgICAgICtzaHV0ZG93bigpXG4gICAgfVxuICAgIGNsYXNzIFN5bmNocm9uaXplcntcbiAgICAgICAgK2FkZF9zb3VyY2VzKG4pXG4gICAgICAgICtyZW1vdmVfc291cmNlcyhuKVxuICAgICAgICArYWRkX3NvdXJjZSgpXG4gICAgICAgICtyZW1vdmVfc291cmNlKClcbiAgICAgICAgK3Jlc2V0KClcbiAgICAgICAgK3N5bmMoZGF0YSwgc3QsIGV0LCBkYXRhX2lkKVxuICAgIH1cbiAgICBjbGFzcyBQcm9jZXNzb3J7XG4gICAgICAgICtzaHV0ZG93bigpXG4gICAgfVxuICAgIGNsYXNzIFN0cmVhbXtcbiAgICAgICAgK2dlbmVyYXRlKCkgcGQuRGF0YWZyYW1lXG4gICAgICAgICtnZXRfc3RhdHVzKCkgU3RyZWFtLlN0YXR1c1xuICAgICAgICArc3RhcnQoc3RhcnRfdGltZSlcbiAgICAgICAgK3N0b3AoKVxuICAgIH1cbiAgICBjbGFzcyBTZWdtZW50b3J7XG4gICAgICAgICtzZXRfcmVmX3RpbWUodHMpXG4gICAgICAgICtyZXNldCgpXG4gICAgICAgICtzZWdtZW50KGRhdGEpIHBkLkRhdGFmcmFtZSwgZGljdFxuICAgIH1cbiAgICBjbGFzcyBTY2hlZHVsZXJ7XG4gICAgICAgICtyZXNldCgpXG4gICAgICAgICtjbG9zZSgpXG4gICAgICAgICtzdWJtaXQoZnVuYywgYXJncywga3dhcmdzKVxuICAgICAgICArZ2V0X2FsbF9yZW1haW5pbmdfcmVzdWx0cygpXG4gICAgICAgICtnZXRfcmVzdWx0KHRpbWVvdXQpIHR1cGxlXG4gICAgfVxuICAgICAgICAiLCJtZXJtYWlkIjp7InRoZW1lIjoibmV1dHJhbCJ9fQ)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiY2xhc3NEaWFncmFtXG4gICAgTm9kZSBvLS0gT3BlcmF0b3JcbiAgICBPcGVyYXRvciA8fC0tIFNlZ21lbnRvclxuXHRPcGVyYXRvciA8fC0tIEdlbmVyYXRvclxuICAgIE9wZXJhdG9yIDx8LS0gU3luY2hyb25pemVyXG4gICAgT3BlcmF0b3IgPHwtLSBQcm9jZXNzb3JcbiAgICBPcGVyYXRvciA8fC0tIFBpcGVsaW5lXG4gICAgU3RyZWFtIG8tLSBTZWdtZW50b3JcbiAgICBTdHJlYW0gby0tIEdlbmVyYXRvclxuICAgIFBpcGVsaW5lIG8tLSBTdHJlYW1cbiAgICBQaXBlbGluZSBvLS0gU3luY2hyb25pemVyXG4gICAgUGlwZWxpbmUgby0tIFByb2Nlc3NvclxuICAgIFByb2Nlc3NvciBvLS0gU2NoZWR1bGVyXG4gICAgY2xhc3MgTm9kZXtcbiAgICAgICAgK2dldF9uYW1lKCkgc3RyXG4gICAgICAgICtnZXRfdHlwZSgpIE8uVHlwZVxuICAgICAgICArZ2V0X29wKCkgT3BlcmF0b3JcbiAgICAgICAgK2dldF9zdGF0dXMoKSBPLlN0YXR1c1xuICAgICAgICArc3RhcnQoKVxuICAgICAgICArc3RvcCgpXG4gICAgICAgICtjb25zdW1lKHBhY2spXG4gICAgICAgICtwcm9kdWNlKClcbiAgICB9XG4gICAgY2xhc3MgT3BlcmF0b3J7XG4gICAgICAgICtydW4oKSpcbiAgICAgICAgK3NldF9jb250ZXh0KClcbiAgICAgICAgK3N0b3AoKVxuICAgICAgICArZ2V0X3Jlc3VsdCgpIHBkLkRhdGFmcmFtZSwgZGljdFxuICAgIH1cbiAgICBjbGFzcyBQaXBlbGluZXtcbiAgICAgICAgK3N0YXJ0KHN0YXJ0X3RpbWUpXG4gICAgICAgICtzaHV0ZG93bigpXG4gICAgfVxuICAgIGNsYXNzIFN5bmNocm9uaXplcntcbiAgICAgICAgK2FkZF9zb3VyY2VzKG4pXG4gICAgICAgICtyZW1vdmVfc291cmNlcyhuKVxuICAgICAgICArYWRkX3NvdXJjZSgpXG4gICAgICAgICtyZW1vdmVfc291cmNlKClcbiAgICAgICAgK3Jlc2V0KClcbiAgICAgICAgK3N5bmMoZGF0YSwgc3QsIGV0LCBkYXRhX2lkKVxuICAgIH1cbiAgICBjbGFzcyBQcm9jZXNzb3J7XG4gICAgICAgICtzaHV0ZG93bigpXG4gICAgfVxuICAgIGNsYXNzIFN0cmVhbXtcbiAgICAgICAgK2dlbmVyYXRlKCkgcGQuRGF0YWZyYW1lXG4gICAgICAgICtnZXRfc3RhdHVzKCkgU3RyZWFtLlN0YXR1c1xuICAgICAgICArc3RhcnQoc3RhcnRfdGltZSlcbiAgICAgICAgK3N0b3AoKVxuICAgIH1cbiAgICBjbGFzcyBTZWdtZW50b3J7XG4gICAgICAgICtzZXRfcmVmX3RpbWUodHMpXG4gICAgICAgICtyZXNldCgpXG4gICAgICAgICtzZWdtZW50KGRhdGEpIHBkLkRhdGFmcmFtZSwgZGljdFxuICAgIH1cbiAgICBjbGFzcyBTY2hlZHVsZXJ7XG4gICAgICAgICtyZXNldCgpXG4gICAgICAgICtjbG9zZSgpXG4gICAgICAgICtzdWJtaXQoZnVuYywgYXJncywga3dhcmdzKVxuICAgICAgICArZ2V0X2FsbF9yZW1haW5pbmdfcmVzdWx0cygpXG4gICAgICAgICtnZXRfcmVzdWx0KHRpbWVvdXQpIHR1cGxlXG4gICAgfVxuICAgICAgICAiLCJtZXJtYWlkIjp7InRoZW1lIjoibmV1dHJhbCJ9fQ)


=== "Flexible computational graph"
    | Building block | Functionality | Child classes | Examples    |
    |----------------|---------------|---------------|-------------|
    | Node           | Coming soon   | Coming soon   | Coming soon |
    | Operator       | Coming soon   | Coming soon   | Coming soon |

=== "Unified data interface"
    | Building block | Functionality | Child classes | Examples    |
    |----------------|---------------|---------------|-------------|
    | Stream         | Coming soon   | Coming soon   | Coming soon |
    | Segmentor      | Coming soon   | Coming soon   | Coming soon |
    | Generator      | Coming soon   | Coming soon   | Coming soon |

=== "Common data processing operators"
    | Building block | Functionality | Child classes | Examples    |
    |----------------|---------------|---------------|-------------|
    | Pipeline       | Coming soon   | Coming soon   | Coming soon |
    | Synchronizer   | Coming soon   | Coming soon   | Coming soon |
    | Scheduler      | Coming soon   | Coming soon   | Coming soon |
    | Processor      | Coming soon   | Coming soon   | Coming soon |

## Data processing module

Data processing module provides mathmatical or transformation functions and classes to compute features or transform sensory or annotation data. This module is organized by data types. Currently it only supports one data type `accel` (raw accelerometer data), but more will be added in the future.

| Data type | Features    | Transformations | Others      |
|-----------|-------------|-----------------|-------------|
| `accel`   | Coming soon | Coming soon     | Coming soon |