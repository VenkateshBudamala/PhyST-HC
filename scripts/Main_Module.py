# ==============================================================================
## Main Pipeline Module
# ==============================================================================


# create save_dir + archive scripts FIRST
save_dir = create_save_dir_and_save_scripts(
    project_root,
    Loss,
    seq_len_list,
    hidden_trans,
    hidden_gcn,
    learning_rate,
    LR_method,
    transformations
)

# ─── START LOGGING ───
logger = start_console_logging(save_dir)

if variable == 'Q':
    (
        cali_ps,
        vali_ps,
        test_ps,
        edge_index,
        selected_features,
        converted_reaches_default,
        importance_df
    ) = prepare_dataset_pipeline(
        swat_folder_path=swat_folder_path,
        sim_name=sim_name,
        excel_path=excel_path,
        obs_to_reach_mapping=obs_to_reach_mapping,
        SWAT_Model_Dates=SWAT_Model_Dates,
        warm_up_years=warm_up_years,
        final_outlet=final_outlet,
        threshold=threshold,
        save_dir=save_dir,
        device=device,
        k2u_node = k2u_node
    )
        
else:
    (
        cali_ps,
        vali_ps,
        test_ps,
        edge_index,
        selected_features,
        converted_reaches_default,
        importance_df
    ) = prepare_dataset_pipeline(
        swat_folder_path=swat_folder_path,
        sim_name=sim_name,
        excel_path=excel_path,
        obs_to_reach_mapping=obs_to_reach_mapping,
        SWAT_Model_Dates=SWAT_Model_Dates,
        warm_up_years=warm_up_years,
        final_outlet=final_outlet,
        threshold=threshold,
        save_dir=save_dir,
        device=device,
        k2u_node = k2u_node,
        seq_len = seq_len
    )


# --- Main Loop over Sequence Lengths ---
results_summary = []
predictions_all = {}

for seq_len in seq_len_list:
    best_model, scaler, train_losses, val_losses = train_model_for_seq_len(
        seq_len=seq_len,
        cali_ps=cali_ps,
        vali_ps=vali_ps,
        selected_features=selected_features,
        edge_index=edge_index,
        converted_reaches_default=converted_reaches_default,
        device=device,
        save_dir=save_dir,
        ImprovedTransformerGCN=ImprovedTransformerGCN,
        criterion_class=Loss,  # or HydroGraphLoss
        hidden_trans=hidden_trans,
        hidden_gcn=hidden_gcn,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        trans_layers=trans_layers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        min_delta=min_delta,
        transformations = transformations,
        LR_method = LR_method
    )


    # ─── Validation ───
    vali_preds, vali_metrics = predict_and_evaluate_phase(
        model=best_model,
        phase_name="Validation",
        seq_len=seq_len,
        data_list=vali_ps,
        period=vali_period,
        save_dir=save_dir,
        scaler=scaler,
        edge_index=edge_index,
        selected_features=selected_features,
        device=device,
        plot_hydrograph=plot_hydrograph,
        plot_scatter=plot_scatter,
        transformations = transformations
    )
    
    # ─── Testing ───
    test_preds, test_metrics = predict_and_evaluate_phase(
        model=best_model,
        phase_name="Testing",
        seq_len=seq_len,
        data_list=test_ps,
        period=test_period,
        save_dir=save_dir,
        scaler=scaler,
        edge_index=edge_index,
        selected_features=selected_features,
        device=device,
        plot_hydrograph=plot_hydrograph,
        plot_scatter=plot_scatter,
        transformations = transformations
    )
    
    results_summary.append(vali_metrics)
    results_summary.append(test_metrics)


    
# ─── STOP LOGGING ───
stop_console_logging(logger)
print("Console log saved successfully.")

print("Pipeline completed successfully!")