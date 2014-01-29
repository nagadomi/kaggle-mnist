#include "nv_core.h"
#include "nv_io.h"
#include "nv_num.h"

#define KNN_K 3
#define CLASS 10
#define NK 12
#define MK 16
#define DIM  128
#define MARGIN 1.0f
#define PUSH_RATIO 1.0f
#define DELTA 0.1f
#define EPOCH 70

#define LOAD_L 0
//#define TRAIN_M(data) (data->m / 8 * 7)
#define TRAIN_M(data) (2000)

int predict(nv_matrix_t *data, nv_matrix_t *labels, nv_matrix_t *vec, int vec_j)
{
	nv_knn_result_t results[KNN_K];	
	int knn[CLASS] = {0};
	int j, n, max_v, max_i;
	int draws[KNN_K] = {0}, draw_idx = 0, has_draw = 0;
	
	n = nv_knn(results, KNN_K, data, vec, vec_j);
	for (j = 0; j < n; ++j) {
		++knn[NV_MAT_VI(labels, results[j].index, 0)];
	}
	max_v = max_i= 0;
	for (j = 0; j < CLASS; ++j) {
		if (max_v < knn[j]) {
			max_v = knn[j];
			max_i = j;
		}
	}
	for (j = 0; j < CLASS; ++j) {
		if (max_i != j && max_v == knn[j]) {
			draws[draw_idx++] = j;
			has_draw = 1;
		}
	}
	if (has_draw) {
		float min_dist = FLT_MAX;
		int k;
		for (j = 0; j < KNN_K; ++j) {
			for (k = 0; k < draw_idx; ++k) {
				if (draws[k] == NV_MAT_VI(labels, results[j].index, 0)) {
					if (min_dist > results[j].dist) {
						min_dist = results[j].dist;
						max_i = NV_MAT_VI(labels, results[j].index, 0);
					}
				}
			}
		}
	}
	return max_i;
}

int main(void)
{
	nv_matrix_t *data = nv_load_matrix_bin("../data/train_data.mat");
	nv_matrix_t *labels = nv_load_matrix_bin("../data/train_labels.mat");
	nv_matrix_t *train_data = nv_matrix_alloc(data->n, TRAIN_M(data));
	nv_matrix_t *train_labels = nv_matrix_alloc(labels->n, TRAIN_M(data));
	nv_matrix_t *test_data = nv_matrix_alloc(data->n, data->m - train_data->m);
	nv_matrix_t *test_labels = nv_matrix_alloc(labels->n, labels->m - train_labels->m);
	nv_matrix_t *l;
	nv_matrix_t *train_lmca = nv_matrix_alloc(DIM, train_data->m);
	nv_matrix_t *test_lmca = nv_matrix_alloc(DIM, test_data->m);
	int i, corret;

	nv_vector_normalize_all(data);
	
	nv_srand(11);
	nv_dataset(data, labels,
			   train_data, train_labels,
			   test_data, test_labels);
	
	printf("train: %d, test: %d, %ddim\n",
		   train_data->m,
		   test_data->m,
		   train_data->n);
#if !LOAD_L
	l = nv_matrix_alloc(train_data->n, DIM);	
	nv_lmca_progress(1);
	nv_lmca(l, train_data, train_labels, NK, MK, MARGIN, PUSH_RATIO, DELTA, EPOCH);
	nv_save_matrix_bin("lmca_l.mat", l);
#else
	l = nv_load_matrix_bin("lmca_l.mat");
#endif
	for (i = 0; i < train_data->m; ++i) {
		nv_lmca_projection(train_lmca, i, l, train_data, i);
	}
	for (i = 0; i < test_data->m; ++i) {
		nv_lmca_projection(test_lmca, i, l, test_data, i);
	}
	printf("end LMCA Projection\n");
	
	corret = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (predict(train_lmca, train_labels, test_lmca, i) == NV_MAT_VI(test_labels, i, 0)) {
			++corret;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)corret / test_data->m * 100.0f,
		   corret, test_data->m);

	nv_matrix_free(&data);
	nv_matrix_free(&labels);
	nv_matrix_free(&train_data);
	nv_matrix_free(&train_labels);
	nv_matrix_free(&test_data);
	nv_matrix_free(&test_labels);
	nv_matrix_free(&train_lmca);
	nv_matrix_free(&test_lmca);
	nv_matrix_free(&l);
	
	return 0;
}
