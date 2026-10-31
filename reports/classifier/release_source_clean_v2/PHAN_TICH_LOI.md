# Phân tích lỗi sáu run source-clean: baseline và context mới

Phân tích chỉ đọc dự đoán, metric, confusion, provenance và log validation theo epoch đã lưu. Không train, không chạy inference, không đọc/chấm test hoặc final test, không dùng GPU và không sửa production. “Đúng/sai” trong báo cáo luôn tính theo nhãn tham chiếu đóng băng, chưa phải phán xử độc lập.

## Kết quả quan sát được

Context mới có hai run suy giảm mạnh ở seed 123 và 2026, còn seed 42 tốt hơn baseline cùng seed về macro-F1. Đây là hiện tượng đầu ra phụ thuộc run/seed; dữ liệu hiện có chưa xác định cơ chế gây ra.

| Seed | Baseline macro-F1 | Context macro-F1 | Chênh lệch C−B | Baseline accuracy | Context accuracy |
| --- | --- | --- | --- | --- | --- |
| 123 | 0.610384 | 0.308909 | -0.301475 | 0.735004 | 0.596811 |
| 42 | 0.605275 | 0.630019 | +0.024744 | 0.735763 | 0.749431 |
| 2026 | 0.632077 | 0.226982 | -0.405095 | 0.760061 | 0.528474 |

Sáu file prediction có cùng 1.317 hàng theo cùng thứ tự, 29 tài liệu; text, label/gold, vị trí và hai neighbors giống nhau. Giữa hai biến thể chỉ Section thay đổi: 1.054/1.317 hàng validation và 2.736/3.942 hàng train. Section trống giảm từ 1.045 xuống 26 ở validation; 138 Section mới dài hơn 120 ký tự. Confusion và macro-F1/accuracy được tính lại từ CSV và khớp số đã lưu.

## Câu baseline đúng nhưng context sai, và chiều ngược lại

Số đếm dưới đây là trên cùng hàng và cùng gold; không gộp các dự đoán sai thành đúng.

| Seed | Cả hai đúng | B đúng/C sai | B sai/C đúng | Cả hai sai |
| --- | --- | --- | --- | --- |
| 123 | 655 | 313 | 131 | 218 |
| 42 | 888 | 81 | 99 | 249 |
| 2026 | 591 | 410 | 105 | 211 |

Các mất/được theo gold, ghi dạng B đúng/C sai → B sai/C đúng:

| Seed | OBJECTIVE | REQUIREMENT | ACTIVITY | EXPECTED_RESULT | OTHER |
| --- | --- | --- | --- | --- | --- |
| 123 | 50 → 5 | 43 → 0 | 57 → 28 | 46 → 2 | 117 → 96 |
| 42 | 3 → 1 | 12 → 8 | 18 → 8 | 3 → 19 | 45 → 63 |
| 2026 | 31 → 11 | 51 → 0 | 134 → 5 | 41 → 0 | 153 → 89 |

## Class collapse và nhầm nhãn quan sát được

Seed 123 và 2026 của context không dự đoán REQUIREMENT lần nào, nên recall/F1 REQUIREMENT bằng 0. Đây là class collapse ở đầu ra dự đoán đã lưu; chưa phải lời giải thích nguyên nhân huấn luyện. Seed 2026 chỉ có 8 dự đoán EXPECTED_RESULT và cả 8 đều sai, tức 0/97 câu gold EXPECTED_RESULT được nhận đúng.

| Run | OBJECTIVE | REQUIREMENT | ACTIVITY | EXPECTED_RESULT | OTHER |
| --- | --- | --- | --- | --- | --- |
| baseline_seed_123 | 153 | 83 | 258 | 109 | 714 |
| context_seed_123 | 194 | 0 | 268 | 79 | 776 |
| baseline_seed_42 | 135 | 86 | 287 | 100 | 709 |
| context_seed_42 | 102 | 74 | 255 | 144 | 742 |
| baseline_seed_2026 | 113 | 95 | 282 | 89 | 738 |
| context_seed_2026 | 313 | 0 | 79 | 8 | 917 |

Bảng confusion tập trung vào hai cặp dễ nhầm; gold → predicted là hướng đọc.

| Run | REQ → ACT | ACT → REQ | ER → OTHER | OTHER → ER | ER đúng | REQ đúng |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_seed_123 | 24 | 8 | 9 | 40 | 55 | 43 |
| context_seed_123 | 48 | 0 | 22 | 31 | 11 | 0 |
| baseline_seed_42 | 32 | 11 | 12 | 44 | 46 | 41 |
| context_seed_42 | 29 | 10 | 17 | 59 | 62 | 37 |
| baseline_seed_2026 | 25 | 9 | 17 | 34 | 41 | 51 |
| context_seed_2026 | 6 | 0 | 57 | 6 | 0 | 0 |

Ở context123, 89 câu REQUIREMENT bị chia thành ACTIVITY 48, OTHER 29, EXPECTED_RESULT 7 và OBJECTIVE 5. Context2026 chia thành OTHER 66, OBJECTIVE 17 và ACTIVITY 6. Nhầm REQUIREMENT–ACTIVITY là một phần của lỗi; sự biến mất toàn bộ lớp REQUIREMENT là vấn đề lớn hơn.

EXPECTED_RESULT–OTHER không giải thích toàn bộ suy giảm. Context123 còn chuyển 41/97 EXPECTED_RESULT sang ACTIVITY và 23 sang OBJECTIVE; context2026 chuyển 34 sang OBJECTIVE. Context42 tăng số ER đúng từ 46 lên 62 và tăng F1 ER, nhưng vẫn tăng ER → OTHER từ 12 lên 17 và OTHER → ER từ 44 lên 59. Vì vậy một cặp confusion xấu đi vẫn có thể đi cùng macro-F1 chung tăng.

OTHER chiếm 843/1.317 câu (64,01%); accuracy có thể che mức mất cân bằng giữa các lớp. Context123 accuracy vẫn 0,596811 trong khi macro-F1 chỉ 0,308909.

## Section mới, neighbors và giới hạn quy nguyên nhân

Các input của từng hàng được bảo toàn ngoài Section; mô hình của hai biến thể đã được huấn luyện riêng. Không có phép đối chứng giữ nguyên một model rồi chỉ thay Section trong phân tích này. Vì Section được đổi ở cả train và validation, chưa tách được tác động của thay đổi input inference, quá trình tối ưu và chọn checkpoint.

| Seed | Nhóm hàng | N | B đúng/C sai | B sai/C đúng |
| --- | --- | --- | --- | --- |
| 123 | Section đổi | 1054 | 234 | 101 |
| 123 | Section giữ nguyên | 263 | 79 | 30 |
| 42 | Section đổi | 1054 | 67 | 79 |
| 42 | Section giữ nguyên | 263 | 14 | 20 |
| 2026 | Section đổi | 1054 | 314 | 88 |
| 2026 | Section giữ nguyên | 263 | 96 | 17 |

Ngay ở 263 câu Section giữ nguyên, context123 và context2026 vẫn mất 79 và 96 dự đoán đúng của baseline. Các case E03, E14, E15 cho thấy lỗi xảy ra khi cả text, Section, vị trí và neighbors của câu giữ nguyên. Không thể gán mọi lỗi trực tiếp cho Section mới ở câu validation.

Input encoder trong snapshot chỉ dùng 600 ký tự đầu text; 120 ký tự cuối Section; 150 ký tự cuối Previous và 150 ký tự đầu Next, rồi tokenizer cắt longest_first ở 256 token. Báo cáo giữ nguyên toàn bộ text/neighbors đã lưu để người đọc kiểm tra; các trường encoder_* trong JSON mô tả phần ký tự trước cắt token. Chưa kiểm tra tokenization của từng câu và không kết luận mất thông tin do cắt token gây ra lỗi.

Các Section chứa ký hiệu công thức (E06) hoặc tên có thể gợi nhãn (E11) là dấu hiệu để rà nguồn. Chưa đối chiếu PDF gốc, chưa xác minh lỗi OCR/tách câu/heading và chưa phán xử lại gold. Mọi case chỉ có phân loại nguyên nhân ở mức observed_model_output; các nguyên nhân context/PDF/nhãn tham chiếu đều unverified.

## Dấu vết từ validation theo epoch đã lưu

Các run context yếu đã có điểm validation thấp ngay trong log Trainer:

| Run | Best macro-F1 trong Trainer | Checkpoint chọn | Macro-F1 saved predictions |
| --- | --- | --- | --- |
| baseline_seed_123 | 0.604416 | checkpoint-1235 | 0.610384 |
| context_seed_123 | 0.308966 | checkpoint-1235 | 0.308909 |
| baseline_seed_42 | 0.607631 | checkpoint-1235 | 0.605275 |
| context_seed_42 | 0.639175 | checkpoint-988 | 0.630019 |
| baseline_seed_2026 | 0.633120 | checkpoint-1235 | 0.632077 |
| context_seed_2026 | 0.221054 | checkpoint-1235 | 0.226982 |

Best Trainer context123 là 0,308966 và context2026 là 0,221054, gần mức yếu ở prediction đã lưu. Vì vậy chưa có bằng chứng suy giảm chỉ xuất hiện lúc export hoặc do khác precision/runtime. Log không xác định được nguyên nhân tối ưu; không suy ra learning rate, loss weighting hoặc Section sai là nguyên nhân chỉ từ các con số này.

## 57 đề xuất nhãn không phải ground truth của study này

Manifest revision lịch sử ghi 68 đề xuất đổi nhãn và 57 thay đổi hiệu lực trên tập train của study nhãn trước; label_origin là user_supplied; independence_not_verified; not_expert_ground_truth. Không nên gọi 57 mục này là 57 nhãn chuyên gia đã xác nhận. Study source-clean sáu run hiện tại sử dụng nhãn gốc đóng băng: cả sáu provenance ghi use_label_overlay=false và label_origin="original frozen labels; no proposed AI changes applied"; experiment_plan ghi ai_labels_applied=false. Các đề xuất đó không được dùng để sửa gold hay gán nguyên nhân nhãn sai trong báo cáo này.

## 15 case minh họa đầy đủ

Đây là mẫu chọn có chủ đích để minh họa các dạng lỗi và ca được cứu, không phải mẫu ngẫu nhiên hoặc tỷ lệ lỗi. Các nhãn dưới đây giữ nguyên tên trong artifact. B = baseline, C = context; mỗi case có đủ sáu dự đoán. Số hàng validation là 1-based, không tính header CSV. Previous/Next là toàn văn hàng lân cận đã lưu và không thay giữa sáu run.

### E01 — PSAI017 — hàng 60

**Text đầy đủ:** User can mint Dataset to NFT with permitted number of keys

**Gold đóng băng:** REQUIREMENT

**Section cũ:** (trống)

**Section mới:** Project Description > Value

**Previous:** User annotated the dataset

**Next:** User can share the NFT via IPFS and backup to Filecoin network.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | REQUIREMENT | ACTIVITY |
| 42 | REQUIREMENT | REQUIREMENT |
| 2026 | REQUIREMENT | OTHER |

**Nhận xét có bằng chứng:** C123 chuyển REQUIREMENT thành ACTIVITY; C2026 chuyển thành OTHER; C42 giữ đúng. Text và hai câu lân cận giống nhau ở cả sáu run. Section từ trống sang Value là thay đổi quan sát được, chưa chứng minh đây là nguyên nhân.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E02 — PSAI028 — hàng 137

**Text đầy đủ:** the file hash should cover how to extend the current CID/multihash to support zkSNARK-friendly hash functions if needed

**Gold đóng băng:** REQUIREMENT

**Section cũ:** (trống)

**Section mới:** Project Description > Deliverables > Development Roadmap

**Previous:** Specifically:

**Next:** the state retrieval protocol should work with today's Filecoin protocol

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | REQUIREMENT | ACTIVITY |
| 42 | REQUIREMENT | REQUIREMENT |
| 2026 | REQUIREMENT | OTHER |

**Nhận xét có bằng chứng:** Cụm should và mô tả ràng buộc nằm trong text; B123/B42/B2026 và C42 đều dự đoán REQUIREMENT. C123 ra ACTIVITY, C2026 ra OTHER. Không suy ra nhãn tham chiếu sai chỉ từ hai run thất bại.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E03 — PSAI137 — hàng 1202

**Text đầy đủ:** Synchronizing the communication using MPI\_Wait or MPI\_Test functions is required to ensure data consistency.

**Gold đóng băng:** REQUIREMENT

**Section cũ:** PROPOSAL FOR GSoC 2023 > Title: Multi-GPU model execution Organization : gprMax > Personal Information > Inter-GPU Communication:

**Section mới:** PROPOSAL FOR GSoC 2023 > Title: Multi-GPU model execution Organization : gprMax > Personal Information > Inter-GPU Communication:

**Previous:** This can be done using non-blocking communication functions like MPI\_Isend and MPI\_Irecv, which allows to perform other computations while data is being exchanged.

**Next:** After the simulation is complete, aggregate the results from all GPUs.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | REQUIREMENT | ACTIVITY |
| 42 | ACTIVITY | REQUIREMENT |
| 2026 | REQUIREMENT | ACTIVITY |

**Nhận xét có bằng chứng:** Section, text và neighbors không đổi; C42 sửa ACTIVITY của B42 thành REQUIREMENT, trong khi C123/C2026 ra ACTIVITY. Đây là bằng chứng khác biệt đầu ra giữa model/run ngay cả khi input của câu này giữ nguyên, không phải bằng chứng Section mới gán sai cho câu.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E04 — PSAI077 — hàng 829

**Text đầy đủ:** Because the proposal studies adaptation from Chinesestrong backbones, Chinese retention must be a first-class target rather than an afterthought.

**Gold đóng băng:** REQUIREMENT

**Section cũ:** (trống)

**Section mới:** 2 Theoretical Framing and Related Work > 5 Evaluation Plan

**Previous:** Efficiency metrics such as token fertility, sequence length, vocabulary growth and training or inference cost will be included whenever tokenizer or vocabulary choices are varied.

**Next:** Related work on multilingual MT robustness shows that robustness can transfer across translation directions but depends on perturbation type, motivating direction-specific stress tests rather than a single aggregate retention score (Pan et al., 2023).

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | REQUIREMENT | OTHER |
| 42 | OTHER | REQUIREMENT |
| 2026 | REQUIREMENT | OBJECTIVE |

**Nhận xét có bằng chứng:** C42 sửa lỗi OTHER của B42; C123 vẫn ra OTHER và C2026 ra OBJECTIVE. Text có must nhưng một tín hiệu từ vựng không bảo đảm model chọn REQUIREMENT. Cần người rà nhãn nếu muốn thay nhãn; chưa có phán xử ở đây.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E05 — PSAI028 — hàng 125

**Text đầy đủ:** If Filecoin can interpolate with Ethereum, it can provide Ethereum layer-2 with lower cost, and it can also increase the demands of storage in the Filecoin network, which benefits the coins, miners, and the ecosystem in general.

**Gold đóng băng:** EXPECTED_RESULT

**Section cũ:** (trống)

**Section mới:** Project Description > Examples of technical challenges:

**Previous:** Needs to examine whether Starkware's FRI is better suit for this purpose.

**Next:** This research is better done sooner than later, as if any protocol change is indeed necessary for this purpose, it is better to be known soon.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | EXPECTED_RESULT | OTHER |
| 42 | EXPECTED_RESULT | EXPECTED_RESULT |
| 2026 | EXPECTED_RESULT | OBJECTIVE |

**Nhận xét có bằng chứng:** Ba baseline và C42 đúng EXPECTED_RESULT; C123 chuyển sang OTHER, C2026 sang OBJECTIVE. Tương quan với Section mới về technical challenges chưa chứng minh Section sai hoặc gây lỗi.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E06 — PSAI064 — hàng 569

**Text đầy đủ:** Fortunately, this affects the development almost not at all, and ultimately we will prove that the value relation can be ignored (Lemma 4.5).

**Gold đóng băng:** EXPECTED_RESULT

**Section cũ:** (trống)

**Section mới:** Γ `<sup>I</sup> M : σ e > 4 Contextual Equivalence

**Previous:** This comes into play in the compatibility rule for polymorphic functions because of the value restriction.

**Next:** Although we define consistency in terms of module termination, it is easy to show that a consistent, compatible relation also preserves term termination.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | EXPECTED_RESULT | OTHER |
| 42 | EXPECTED_RESULT | EXPECTED_RESULT |
| 2026 | EXPECTED_RESULT | OTHER |

**Nhận xét có bằng chứng:** C123/C2026 ra OTHER còn C42 và ba baseline đúng. Section mới chứa ký hiệu công thức và tiêu đề Contextual Equivalence; đây là bất thường văn bản có thể đưa vào hàng chờ rà nguồn, chưa xác minh lỗi PDF/OCR hay lỗi gán heading.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E07 — PSAI033 — hàng 198

**Text đầy đủ:** This feature-set allows any software to be distributed with NEAR + IPFS/Filecoin, not just the core protocol tooling, so any developer in the ecosystem can leverage this for their packages.

**Gold đóng băng:** EXPECTED_RESULT

**Section cũ:** (trống)

**Section mới:** Project Description

**Previous:** This further decentralizes the process for Nearcore Planned Updates.

**Next:** A NEAR-native, AssemblyScript package registry smart contract that provides role-based access control, package namespacing, and versioning.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | EXPECTED_RESULT | ACTIVITY |
| 42 | EXPECTED_RESULT | EXPECTED_RESULT |
| 2026 | EXPECTED_RESULT | OTHER |

**Nhận xét có bằng chứng:** Ba baseline và C42 đúng; C123 ra ACTIVITY, C2026 ra OTHER. Text mô tả khả năng/phạm vi sử dụng trong khi câu kế tiếp bắt đầu một sản phẩm bàn giao. Chưa xác định phần ngữ cảnh nào chi phối model.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E08 — PSAI033 — hàng 200

**Text đầy đủ:** A Golang SDK for the RPC APIs will be developed to enable Valist to support the protocol.

**Gold đóng băng:** EXPECTED_RESULT

**Section cũ:** (trống)

**Section mới:** Project Description > Deliverables

**Previous:** A NEAR-native, AssemblyScript package registry smart contract that provides role-based access control, package namespacing, and versioning.

**Next:** This can become a near-api-go implementation, and has the potential to bring many Go developers into the NEAR ecosystem.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | ACTIVITY | ACTIVITY |
| 42 | ACTIVITY | EXPECTED_RESULT |
| 2026 | ACTIVITY | OTHER |

**Nhận xét có bằng chứng:** Ba baseline ra ACTIVITY, C42 ra EXPECTED_RESULT theo nhãn đóng băng; C123 giữ ACTIVITY, C2026 ra OTHER. Cấu trúc will be developed có thể khiến người đọc cân nhắc hai loại nhãn, nhưng đây là yêu cầu rà nhãn có thể có, không phải kết luận nhãn gold sai.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E09 — PSAI028 — hàng 135

**Text đầy đủ:** The deliverables would sufficiently cover the state-of-the-arts techniques regarding zk-Rollup and zkSNARK, so it can serve as a good systematization of knowledge.

**Gold đóng băng:** EXPECTED_RESULT

**Section cũ:** (trống)

**Section mới:** Project Description > Deliverables > Development Roadmap

**Previous:** In total 14 weeks, slightly exceeding 3 months.

**Next:** Specifically:

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | EXPECTED_RESULT | OTHER |
| 42 | OTHER | EXPECTED_RESULT |
| 2026 | OTHER | OTHER |

**Nhận xét có bằng chứng:** C42 sửa OTHER của B42 thành EXPECTED_RESULT; C123 chuyển dự đoán đúng của B123 sang OTHER, B2026/C2026 cùng OTHER. Cùng Section mới có cả kết quả cải thiện lẫn suy giảm giữa các seed.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E10 — PSAI017 — hàng 45

**Text đầy đủ:** Adding comprehensive and consistent tags is a key part of developing a training dataset for machine learning.

**Gold đóng băng:** OTHER

**Section cũ:** (trống)

**Section mới:** Project Description

**Previous:** This metadata usually takes the form of tags, which can be added to any type of data, including text, images, and video.

**Next:** Deep learning data annotation is the next 100Billion dollar market, it is a pretty mature market in China and india.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | OTHER | OTHER |
| 42 | OTHER | EXPECTED_RESULT |
| 2026 | OTHER | OTHER |

**Nhận xét có bằng chứng:** C42 ra EXPECTED_RESULT dù ba baseline và C123/C2026 đều OTHER. Đây là một lỗi mới của run context tốt nhất; không thể mô tả context42 là cải thiện mọi câu.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E11 — PSAI052 — hàng 290

**Text đầy đủ:** With a sharply decreasing weight function, we can find rank-maximal matchings, which match the maximum number of agents to their first choice, and subject to this, the maximum number to their second choice, and so on

**Gold đóng băng:** OTHER

**Section cũ:** (trống)

**Section mới:** 1 Introduction > 2.1 Static Objective Functions

**Previous:** A matching algorithm selects a matching based on the preferences of agents over outcomes

**Next:** There has been some nice recent work on finding faster algorithms for these objectives

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | OTHER | OBJECTIVE |
| 42 | OTHER | EXPECTED_RESULT |
| 2026 | OTHER | OBJECTIVE |

**Nhận xét có bằng chứng:** Ba baseline đúng OTHER; C123/C2026 ra OBJECTIVE, C42 ra EXPECTED_RESULT. Text mô tả cơ chế matching trong phần Static Objective Functions. Từ objective trong heading là dấu hiệu cần xem xét, chưa chứng minh gây lệch nhãn.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E12 — PSAI056 — hàng 386

**Text đầy đủ:** A computer vision (CV) module will be implemented to detect the location of physical checker pieces placed on the screen.

**Gold đóng băng:** ACTIVITY

**Section cũ:** (trống)

**Section mới:** Computer Vision

**Previous:** Block Diagram

**Next:** The module interacts with inputs from the NTSC camera and the central game logic module.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | ACTIVITY | OBJECTIVE |
| 42 | ACTIVITY | ACTIVITY |
| 2026 | ACTIVITY | OTHER |

**Nhận xét có bằng chứng:** Ba baseline và C42 đúng ACTIVITY; C123 ra OBJECTIVE, C2026 ra OTHER. Text về module CV và neighbors được giữ nguyên; Section được bổ sung Computer Vision. Chưa có bằng chứng lỗi tách PDF.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E13 — PSAI028 — hàng 111

**Text đầy đủ:** This project is to explore the technical designs for it.

**Gold đóng băng:** OBJECTIVE

**Section cũ:** (trống)

**Section mới:** Project Description

**Previous:** Putting data on layer-1, which is the standard solution for data availability, is very expensive.

**Next:** The goal is to enable a zk-Rollup layer-2 to prove that certain data has been stored on Filecoin.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | OBJECTIVE | OTHER |
| 42 | OBJECTIVE | OBJECTIVE |
| 2026 | OBJECTIVE | OTHER |

**Nhận xét có bằng chứng:** Ba baseline và C42 đúng OBJECTIVE; C123/C2026 ra OTHER. Câu kế tiếp có The goal; neighbors không đổi trong cặp so sánh. Không có thí nghiệm riêng bỏ Section hoặc neighbors trong phân tích này.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E14 — PSAI102 — hàng 903

**Text đầy đủ:** RQ3: What is the impact of diverse raters on the amount of data needed for reproducibility?

**Gold đóng băng:** OBJECTIVE

**Section cũ:** (trống)

**Section mới:** (trống)

**Previous:** RQ2: How do we optimize the allocation of a fixed human annotation budget for reproducible evaluation?

**Next:** In the case of LLMs, humans continue to play a vital role in all stages of their development.

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | OBJECTIVE | OTHER |
| 42 | OBJECTIVE | OBJECTIVE |
| 2026 | OBJECTIVE | EXPECTED_RESULT |

**Nhận xét có bằng chứng:** Section vẫn trống trong cả hai biến thể. Ba baseline và C42 đúng OBJECTIVE; C123 ra OTHER, C2026 ra EXPECTED_RESULT. Loại trừ phát biểu rằng lỗi của riêng câu này đến từ Section validation bị thay; khác biệt model sau huấn luyện vẫn chưa được phân rã.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

### E15 — PSAI137 — hàng 1190

**Text đầy đủ:** This approach allows to distribute the model across multiple GPUs, enabling large-scale simulations without being limited by the memory of a single GPU.

**Gold đóng băng:** EXPECTED_RESULT

**Section cũ:** PROPOSAL FOR GSoC 2023 > Title: Multi-GPU model execution Organization : gprMax > Personal Information > Implementation:

**Section mới:** PROPOSAL FOR GSoC 2023 > Title: Multi-GPU model execution Organization : gprMax > Personal Information > Implementation:

**Previous:** One of the ways of implementing this is to use a hybrid approach that combines MPI (Message Passing Interface) with CUDA-aware MPI for domain decomposition and memory sharing.

**Next:** Here's a general flowchart of the process:

| Seed | Baseline | Context |
| --- | --- | --- |
| 123 | EXPECTED_RESULT | ACTIVITY |
| 42 | EXPECTED_RESULT | EXPECTED_RESULT |
| 2026 | EXPECTED_RESULT | OTHER |

**Nhận xét có bằng chứng:** Section dài và không đổi; ba baseline/C42 đúng EXPECTED_RESULT, C123 ra ACTIVITY, C2026 ra OTHER. Đây là lỗi trên input không đổi, nhất quán với khác biệt model giữa hai biến thể, chưa xác định cơ chế huấn luyện.

**Phân loại nguyên nhân:** đầu ra model và metadata input đã quan sát; nguyên nhân Section/neighbors, PDF/OCR/tách câu và nhãn tham chiếu chưa xác minh.

## Nguồn và khả năng kiểm tra lại

- [JSON đầy đủ: confusion, số đếm cặp, hash nguồn và 15 case](./error_analysis.json).
- [Kế hoạch study](../../research_review/studies/classifier_source_clean_context_study_v1/experiment_plan.json).
- [Snapshot input encoder](../../research_review/studies/classifier_source_clean_context_study_v1/source_snapshot/study_inputs.py).
- [Manifest revision nhãn lịch sử](../../research_review/studies/user_labels_v1/dataset_revision.json).

Các metric, confusion, prediction và provenance của mỗi run nằm trong thư mục experiments của study; đường dẫn và SHA-256 từng nguồn được ghi trong JSON. Phân tích này không tạo nhãn mới hoặc xác nhận nguyên nhân chưa kiểm chứng.
