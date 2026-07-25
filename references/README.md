# Curated References for the STAR-RIS–RSMA TD3 Thesis

This directory contains a reviewer-curated reading list aligned with the checked-in implementation under `star_ris_rsma/`, `scripts/`, `configs/`, `kaggle_runs/`, and `results/`.

## Usage rules

- Use the papers below to explain concepts that are actually implemented in the repository.
- Do not copy a paper's system model into the thesis when it differs from the code. In particular, distinguish SISO from MISO/mMIMO, independent from coupled STAR-RIS phase models, passive from active STAR-RIS, and centralized TD3 from multi-agent MADDPG.
- The final experimental claim remains a **quality–latency trade-off**: AO-SCA attains higher sum-rate, while TD3 provides high QoS reliability at much lower decision latency.
- PDFs are downloaded only from arXiv, institutional repositories, Zenodo, PubMed Central/MDPI, or other explicitly open author manuscripts. Paywalled publisher PDFs are not copied into the repository.
- Canonical citation keys are in `references.bib`; open-access locations are in `open_access_pdfs.tsv`.

## A. RSMA foundations and RIS–RSMA surveys

1. **Mao et al. (2022), “Rate-Splitting Multiple Access: Fundamentals, Survey, and Future Research Trends.”**  
   Use for the core RSMA taxonomy, common/private streams, SIC, QoS and interference-management interpretation. DOI: `10.1109/COMST.2022.3191937`; OA: arXiv `2201.03192`.

2. **Mao, Clerckx, and Li (2018), “Rate-Splitting Multiple Access for Downlink Communication Systems: Bridging, Generalizing and Outperforming SDMA and NOMA.”**  
   Use to explain why RSMA bridges treating interference as noise and fully decoding interference. DOI: `10.1186/s13638-018-1104-7`; OA: arXiv `1710.11018`.

3. **Joudeh and Clerckx (2016), “Sum-Rate Maximization for Linearly Precoded Downlink Multiuser MISO Systems with Partial CSIT: A Rate-Splitting Approach.”**  
   Use for formal sum-rate optimization and RS precoding under imperfect CSIT. DOI: `10.1109/TCOMM.2016.2600671`; OA: arXiv `1602.09028`.

4. **Li et al. (2020), “Rate Splitting for Multi-Antenna Downlink: Precoder Design and Practical Implementation.”**  
   Use for practical precoder design, decoding order, stream selection and implementation complexity. OA: arXiv `2002.07225`.

5. **Clerckx et al. (2021), “Is NOMA Efficient in Multi-Antenna Networks? A Critical Look at Next Generation Multiple Access Techniques.”**  
   Use as a reviewer-quality justification for comparing RSMA with SDMA/NOMA without overstating NOMA. DOI: `10.1109/OJCOMS.2021.3054799`; OA: arXiv `2101.04802`.

6. **Li, Mao, Dizdar, and Clerckx (2022), “Rate-Splitting Multiple Access for 6G—Part III: Interplay with Reconfigurable Intelligent Surfaces.”**  
   Use as the conceptual bridge between the RSMA signal model and RIS-assisted propagation. OA: arXiv `2205.02036`.

7. **Aboumahmoud, Hossain, and Mezghani (2024), “Resource Management in RIS-Assisted Rate Splitting Multiple Access for Next Generation (xG) Wireless Communications: Models, State-of-the-Art, and Future Directions.”**  
   Closest survey to the thesis topic; use for related-work structure, model taxonomy, resource allocation and learning-based methods. OA: arXiv `2404.06604`.

## B. RIS fundamentals and physical-model guardrails

8. **Wu and Zhang (2019), “Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming.”**  
   Canonical joint active/passive beamforming and alternating-optimization reference. DOI: `10.1109/TWC.2019.2936025`; OA: arXiv `1810.03961`.

9. **Huang et al. (2019), “Reconfigurable Intelligent Surfaces for Energy Efficiency in Wireless Communication.”**  
   Use for RIS power-consumption modeling and alternating/sequential resource optimization. DOI: `10.1109/TWC.2019.2922609`; OA: arXiv `1810.06934`.

10. **Basar et al. (2019), “Wireless Communications Through Reconfigurable Intelligent Surfaces.”**  
    Use for the early RIS overview, propagation control and differences from conventional relaying. DOI: `10.1109/ACCESS.2019.2935192`; OA: arXiv `1906.09490`.

11. **Di Renzo et al. (2020), “Smart Radio Environments Empowered by Reconfigurable Intelligent Surfaces: How It Works, State of Research, and Road Ahead.”**  
    Use to separate electromagnetic assumptions from simplified communication-theoretic channel equations. DOI: `10.1109/JSAC.2020.3007211`; OA: arXiv `2004.09352`.

12. **Björnson, Özdogan, and Larsson (2020), “Reconfigurable Intelligent Surfaces: Three Myths and Two Critical Questions.”**  
    Mandatory guardrail against unsupported array-gain, path-loss and scalability claims. DOI: `10.1109/MCOM.001.2000407`; OA: arXiv `2006.03377`.

13. **Wu et al. (2021), “Intelligent Reflecting Surface Aided Wireless Communications: A Tutorial.”**  
    Use for channel models, reflection coefficients, practical phase constraints, channel estimation and deployment. DOI: `10.1109/TCOMM.2021.3051897`; OA: arXiv `2007.02759`.

## C. STAR-RIS models, protocols and optimization

14. **Xu, Liu, Mu, and Dobre (2021), “STAR-RISs: Simultaneous Transmitting and Reflecting Reconfigurable Intelligent Surfaces.”**  
    Use for the basic STAR-RIS hardware concept, full-space coverage and transmission/reflection links. OA: arXiv `2101.09663`.

15. **Mu et al. (2021), “Simultaneously Transmitting And Reflecting (STAR) RIS Aided Wireless Communications.”**  
    Primary reference for energy-splitting, mode-switching and time-switching protocols and their joint optimization. OA: arXiv `2104.01421`.

16. **Liu, Mu, Schober, and Poor (2021), “Simultaneously Transmitting and Reflecting (STAR)-RISs: A Coupled Phase-Shift Model.”**  
    Important model caveat: a passive STAR-RIS may not permit independent transmission and reflection phases. OA: arXiv `2110.02374`.

17. **Wang, Mu, Liu, and Schober (2022), “Coupled Phase-Shift STAR-RISs: A General Optimization Framework.”**  
    Use for KKT-stationary optimization of coupled amplitude/phase coefficients. OA: arXiv `2208.01942`.

18. **Khalid et al. (2022), “Simultaneous Transmitting and Reflecting-Reconfigurable Intelligent Surface in 6G: Design Guidelines and Future Perspectives.”**  
    Use for STAR-RIS design taxonomy, channel/hardware challenges and future-work discussion. OA: arXiv `2212.01097`.

19. **Zhong et al. (2022), “Hybrid Reinforcement Learning for STAR-RISs: A Coupled Phase-Shift Model Based Beamformer.”**  
    Closest STAR-RIS reinforcement-learning precedent for mixed continuous/discrete decisions and coupled phase constraints. OA: arXiv `2205.05029`.

## D. Direct STAR-RIS + RSMA studies

20. **Meng et al. (2024), “Sum-Rate Maximization in STAR-RIS-Assisted RSMA Networks: A PPO-Based Algorithm.”**  
    The most direct learning-based comparator: STAR-RIS + RSMA, state/action/reward design, QoS constraints and PPO. IEEE IoT Journal 11(4), 5667–5680. DOI: `10.1109/JIOT.2023.3309859`. Publisher/institutional metadata only; no unrestricted PDF was identified.

21. **Ge et al. (2024), “A Rate-Splitting Strategy for STAR-RIS-Aided Massive MIMO Systems With Joint Optimization.”**  
    Use for direct STAR-RIS rate-splitting modeling, ES/MS protocols and joint optimization, while noting that its mMIMO model differs from this repo's SISO implementation. IEEE Systems Journal 18(2), 977–988. DOI: `10.1109/JSYST.2024.3398249`; OA accepted manuscript available.

22. **Liu and Zhou (2024), “Ergodic Rate Analysis of Simultaneous Transmitting and Reflecting Reconfigurable Intelligent Surface-Assisted Rate-Splitting Multiple Access Systems Based on Discrete Phase Shifts.”**  
    Use for discrete-versus-continuous STAR-RIS phase analysis and ergodic-rate derivations. Sensors 24(17), 5480. DOI: `10.3390/s24175480`; open access.

23. **Chang et al. (2024), “Covert Communications in STAR-RIS-Aided Rate-Splitting Multiple Access Systems.”**  
    Direct STAR-RIS-RSMA formulation with QoS, power/rate allocation, AO and penalty-SCA. Physical Communication 64, 102342. DOI: `10.1016/j.phycom.2024.102342`; OA preprint: arXiv `2312.01042`.

24. **Maghrebi et al. (2024), “Cooperative Rate Splitting Multiple Access for Active STAR-RIS Assisted Downlink Communications.”**  
    Use as an active/cooperative STAR-RIS extension and future-work boundary, not as the physical model of this passive repo. IEEE Wireless Communications Letters 13(10), 2827–2831. DOI: `10.1109/LWC.2024.3448409`. Metadata only in this library.

25. **Hashempour and Berardinelli (2024), “Secure Rate Splitting in STAR-RIS Assisted Downlink MISO Systems.”**  
    AO/SCA-based secure STAR-RIS-RSMA example with an openly licensed author manuscript. IEEE MeditCom 2024, 529–534. DOI: `10.1109/MeditCom61057.2024.10621324`.

26. **Liu et al. (2025), “STAR-RIS Enabled ISAC Systems With RSMA: Joint Rate Splitting and Beamforming Optimization.”**  
    Recent extension using SDR, MM and sequential rank-one relaxation; use in future-work/ISAC discussion, not to describe the current communications-only environment. OA: arXiv `2411.09154` and institutional accepted manuscript.

## E. TD3, DDPG, MADDPG and SCA foundations

27. **Fujimoto, van Hoof, and Meger (2018), “Addressing Function Approximation Error in Actor-Critic Methods.”**  
    Original TD3 paper: twin critics, delayed policy updates and target-policy smoothing. ICML 2018; OA: arXiv `1802.09477`.

28. **Lillicrap et al. (2015), “Continuous Control with Deep Reinforcement Learning.”**  
    DDPG foundation for deterministic actor-critic learning in continuous action spaces. OA: arXiv `1509.02971`.

29. **Lowe et al. (2017), “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.”**  
    MADDPG foundation and centralized-training/decentralized-execution rationale. NeurIPS 2017; OA: arXiv `1706.02275`.

30. **Scutari, Facchinei, Lampariello, and Song (2017), “Parallel and Distributed Methods for Nonconvex Optimization—Part I: Theory.”**  
    General convergence framework for successive convex approximation; use to justify the mathematical role and limitations of the AO/SCA baseline. OA: arXiv `1410.4754`.

## Recommended mapping to thesis chapters

- **Chapter 1 — Motivation and related work:** 1, 5, 7, 10–13, 18.
- **Chapter 2 — RSMA system theory:** 1–6.
- **Chapter 3 — RIS/STAR-RIS channel and coefficient model:** 8–19.
- **Chapter 4 — Optimization problem and baselines:** 8, 9, 15–17, 23, 25, 30.
- **Chapter 5 — TD3/MADDPG formulation:** 19, 20, 27–29.
- **Chapter 6 — Experimental comparison:** 7, 20–26 plus `results/FINAL_RESULTS_REVIEW.md`.
- **Limitations/future work:** 11–13, 16–18, 24–26.
