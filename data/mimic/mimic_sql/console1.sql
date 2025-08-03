WITH admissions_subset AS (
    SELECT subject_id, hadm_id
    FROM mimiciv_hosp.admissions
    ORDER BY subject_id, hadm_id
    LIMIT 100000
    OFFSET 0 -- 修改 OFFSET 值以提取不同的批次
)
SELECT
    a.hadm_id as hadm_id,
    p.subject_id as subject_id,
    p.gender as gender,
    p.anchor_age as anchor_age,
    d_icd.icd_code as icd_code,
    d_icd.icd_version as icd_version,
    d_icd_diagnoses.long_title as long_title
--    drgcodes.drg_code as drg_code,
--    drgcodes.drg_type as drg_type,
--    drgcodes.description as description,
--    drgcodes.drg_severity as drg_severity,
--    drgcodes.drg_mortality as drg_mortality
--    emar.emar_id as emar_id,
--    emar.emar_seq as emar_seq,
--    emar.medication as medication,
--    emar.event_txt as event_txt,
--    emar_detail.dose_due as dose_due,
--    emar_detail.dose_due_unit as dose_due_unit,
--    emar_detail.product_description as product_description,
--    emar_detail.product_unit as product_unit,
--    emar_detail.product_amount_given as product_amount_given,
--    emar_detail.product_code as product_code,
--    labevent.itemid as itemid,
--    labevent.value as labevent_value,
--    labevent.valuenum as valuenum,
--    labevent.valueuom as valueuom,
--    labevent.ref_range_lower as ref_range_lower,
--    labevent.ref_range_upper as ref_range_upper,
--    labevent.comments as labevent_comments,
--    d_labitems.label as labevent_label,
--    d_labitems.fluid as labevent_fluid,
--    d_labitems.category as labevent_category,
--    microbiologyevents.microevent_id as microevent_id,
--    microbiologyevents.spec_type_desc as microevent_spec_type_desc,
--    microbiologyevents.test_name as microevent_test_name,
--    microbiologyevents.comments as microevent_comments,
--    microbiologyevents.org_name as microevent_org_name,
--    microbiologyevents.ab_name as microevent_ab_name,
--    microbiologyevents.dilution_text as microevent_dilution_text,
--    microbiologyevents.interpretation as microevent_interpretation,

--    omr.chartdate as omr_chartdate,
--    omr.result_name as omr_result_name,
--    omr.result_value as omr_result_value
--    prescriptions.starttime as prescriptions_starttime,
--    prescriptions.stoptime as prescriptions_stoptime,
--    prescriptions.drug_type as prescriptions_drug_type,
--    prescriptions.drug as prescriptions_drug,
--    prescriptions.gsn as prescriptions_gsn,
--    prescriptions.ndc as prescriptions_ndc,
--    prescriptions.prod_strength as prescriptions_prod_strength,
--    prescriptions.form_rx as prescriptions_form_rx,
--    prescriptions.dose_val_rx as prescriptions_dose_val_rx,
--    prescriptions.dose_unit_rx as prescriptions_dose_unit_rx,
--    prescriptions.form_val_disp as prescriptions_form_val_disp,
--    prescriptions.form_unit_disp as prescriptions_form_unit_disp,
--    prescriptions.doses_per_24_hrs as prescriptions_doses_per_24_hrs,
--    prescriptions.route as prescriptions_route
FROM
    admissions_subset a
    JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
    JOIN mimiciv_hosp.diagnoses_icd d_icd ON a.hadm_id = d_icd.hadm_id
    JOIN mimiciv_hosp.d_icd_diagnoses d_icd_diagnoses ON d_icd.icd_code = d_icd_diagnoses.icd_code AND d_icd.icd_version = d_icd_diagnoses.icd_version
--    LEFT JOIN mimiciv_hosp.drgcodes drgcodes ON a.hadm_id = drgcodes.hadm_id
--    JOIN mimiciv_hosp.emar emar ON a.subject_id = emar.subject_id
--    JOIN mimiciv_hosp.emar_detail emar_detail ON emar.emar_id = emar_detail.emar_id
--    JOIN mimiciv_hosp.labevents labevent ON a.hadm_id = labevent.hadm_id
--    JOIN mimiciv_hosp.d_labitems d_labitems ON labevent.itemid = d_labitems.itemid
--    LEFT JOIN mimiciv_hosp.microbiologyevents microbiologyevents ON a.subject_id = microbiologyevents.subject_id
--    LEFT JOIN mimiciv_hosp.omr omr ON a.subject_id = omr.subject_id
--    LEFT JOIN mimiciv_hosp.prescriptions prescriptions ON a.hadm_id = prescriptions.hadm_id

