#include "nv_core.h"
#include "nv_io.h"
#include "nv_ip.h"
#include "nv_num.h"
#include "nv_fv_rectangle_feature.h"

#define HIDDEN_UNIT 1024
#define CLASS 10

#define TRAIN_M(data) (data->m / 8 * 7)
//#define TRAIN_M(data) (2000)

void
validation(const nv_mlp_t *mlp,
		   const nv_matrix_t *test_data,
		   const nv_matrix_t *test_labels)
{
	int i, corret = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (nv_mlp_predict_label(mlp, test_data, i) == NV_MAT_VI(test_labels, i, 0)) {
			++corret;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)corret / test_data->m * 100.0f,
		   corret, test_data->m);
}

void
rectangle_feature(nv_matrix_t *fv, nv_matrix_t *data)
{
	int i;
	for (i = 0; i < data->m; ++i) {
		nv_matrix_t *image = nv_vector_reshape3d(data, i, 1, 32, 32);
		nv_matrix_t *integral = nv_matrix3d_alloc(1, image->rows + 1, image->cols + 1);

		nv_integral(integral, image, 0);
		nv_rectangle_feature(fv, i, integral, 0, 0, 32, 32);

		nv_matrix_free(&integral);
		nv_matrix_free(&image);
	}
	printf("end rectangle_feature\n");
}

int
main(void)
{
	nv_matrix_t *data_src = nv_load_matrix_bin("../data/train_data32.mat");
	nv_matrix_t *data = nv_matrix_alloc(NV_RECTANGLE_FEATURE_N, data_src->m);
	nv_matrix_t *labels = nv_load_matrix_bin("../data/train_labels.mat");
	nv_matrix_t *train_data = nv_matrix_alloc(data->n, TRAIN_M(data));
	nv_matrix_t *train_labels = nv_matrix_alloc(labels->n, TRAIN_M(data));
	nv_matrix_t *test_data = nv_matrix_alloc(data->n, data->m - train_data->m);
	nv_matrix_t *test_labels = nv_matrix_alloc(labels->n, labels->m - train_labels->m);
	nv_matrix_t *zca_u = nv_matrix_alloc(data->n, data->n);
	nv_matrix_t *zca_m = nv_matrix_alloc(data->n, 1);
	nv_matrix_t *sd_m = nv_matrix_alloc(data->n, 1);
	nv_matrix_t *sd_sd = nv_matrix_alloc(data->n, 1);
	nv_mlp_t *mlp = nv_mlp_alloc(data->n, HIDDEN_UNIT, CLASS);
	
	int i;
	float ir, hr;

	rectangle_feature(data, data_src);
	nv_srand(11);
	nv_dataset(data, labels,
			   train_data, train_labels,
			   test_data, test_labels);
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);

	nv_standardize_local_all(train_data, 10.0);
	nv_standardize_local_all(test_data, 10.0);
	printf("local standardize\n");

	nv_zca_train(zca_m, 0, zca_u, train_data, 0.1f);
	nv_zca_whitening_all(train_data, zca_m, 0, zca_u);
	nv_zca_whitening_all(test_data, zca_m, 0, zca_u);
	printf("whitend\n");

	nv_standardize_train(sd_m, 0, sd_sd, 0, train_data, 0.01f);
	nv_standardize_all(train_data, sd_m, 0, sd_sd, 0);
	nv_standardize_all(test_data, sd_m, 0, sd_sd, 0);
	printf("global standardize\n");

	nv_mlp_progress(1);
	nv_mlp_init(mlp, train_data);
	nv_mlp_noise(mlp, 0.2f);
	nv_mlp_dropout(mlp, 0.5f);
	ir = hr = 0.0001f;
	for (i = 0; i < 10; ++i) {
		char file[256];
		nv_mlp_train_ex(mlp, train_data, train_labels, ir, hr,
						i * 100, (1 + i) * 100, 1000);
		validation(mlp, test_data, test_labels);
		
		nv_snprintf(file, sizeof(file), "epoch_%d.mlp", i);
		nv_save_mlp(file, mlp);
	}
	
	nv_matrix_free(&data);
	nv_matrix_free(&data_src);
	nv_matrix_free(&labels);
	nv_matrix_free(&train_data);
	nv_matrix_free(&train_labels);
	nv_matrix_free(&test_data);
	nv_matrix_free(&test_labels);
	nv_matrix_free(&zca_u);
	nv_matrix_free(&zca_m);
	nv_matrix_free(&sd_m);
	nv_matrix_free(&sd_sd);
	nv_mlp_free(&mlp);
	
	return 0;
}
