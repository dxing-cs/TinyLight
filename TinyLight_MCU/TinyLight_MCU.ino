#include "model.h"

void read_feature(const char *s, int16_t *feature_1, int16_t *feature_2, uint8_t dim_feature_1, uint8_t dim_feature_2) {
    memset(feature_1, 0, sizeof(int16_t) * dim_feature_1);
    memset(feature_2, 0, sizeof(int16_t) * dim_feature_2);
    int sign = 1;
    int16_t *feature = feature_1;

    for (uint8_t s_idx = 0; ; ++s_idx) {
        if (s[s_idx] == '-') {
            sign = -1;
        }
        else if (s[s_idx] >= '0' && s[s_idx] <= '9') {
            *feature = (*feature) * 10 + sign * (s[s_idx] - '0');
        }
        else if (s[s_idx] == ',') {
            ++feature;
            sign = 1;
        }
        else if (s[s_idx] == ';') {
            feature = feature_2;
            sign = 1;
        }
        else {
            break;
        }
    }
}

int64_t quan_multiply(int64_t val, int32_t quan_multiplier, uint8_t right_shift) {
    val = val * static_cast<int64_t>(quan_multiplier);
    return val >> right_shift;
}

void linear_relu(
        const int16_t *lhs,
        const int16_t *rhs,
        const int16_t *bias,
        int16_t *result,
        const int32_t quan_multiplier_w,
        const uint8_t right_shift_w,
        const int32_t quan_multiplier_b,
        const uint8_t right_shift_b,
        const uint16_t depth,
        const uint16_t col
) {
    for (uint16_t col_idx = 0; col_idx < col; ++col_idx) {
        uint16_t rhs_flatten_idx = col_idx;
        int64_t quan_matmul = 0;
        for (uint16_t depth_idx = 0; depth_idx < depth; ++depth_idx) {
            quan_matmul += static_cast<int64_t>(lhs[depth_idx]) * static_cast<int64_t>(static_cast<int16_t>(pgm_read_dword_near(rhs + rhs_flatten_idx)));
            rhs_flatten_idx += col;
        }
        quan_matmul = quan_multiply(quan_matmul, quan_multiplier_w, right_shift_w);

        int64_t quan_bias = quan_multiply(static_cast<int64_t>(bias[col_idx]), quan_multiplier_b, right_shift_b);
        quan_matmul = quan_matmul + quan_bias;
        quan_matmul = quan_matmul > 0 ? quan_matmul : 0;
        quan_matmul = quan_matmul < int16_max ? quan_matmul : int16_max;
        result[col_idx] = static_cast<int16_t>(quan_matmul);
    }
}

void linear(
        const int16_t *lhs,
        const int16_t *rhs,
        const int16_t *bias,
        int16_t *result,
        const int32_t quan_multiplier_w,
        const uint8_t right_shift_w,
        const int32_t quan_multiplier_b,
        const uint8_t right_shift_b,
        const uint16_t depth,
        const uint16_t col
) {
    for (uint16_t col_idx = 0; col_idx < col; ++col_idx) {
        uint16_t rhs_flatten_idx = col_idx;
        int64_t quan_matmul = 0;
        for (uint16_t depth_idx = 0; depth_idx < depth; ++depth_idx) {
            quan_matmul += static_cast<int64_t>(lhs[depth_idx]) * static_cast<int64_t>(static_cast<int16_t>(pgm_read_dword_near(rhs + rhs_flatten_idx)));
            rhs_flatten_idx += col;
        }
        quan_matmul = quan_multiply(quan_matmul, quan_multiplier_w, right_shift_w);

        int64_t quan_bias = quan_multiply(static_cast<int64_t>(bias[col_idx]), quan_multiplier_b, right_shift_b);
        quan_matmul = quan_matmul + quan_bias;
        quan_matmul = quan_matmul > int16_min ? quan_matmul : int16_min;
        quan_matmul = quan_matmul < int16_max ? quan_matmul : int16_max;
        result[col_idx] = static_cast<int16_t>(quan_matmul);
    }
}

void add(
        const int16_t *lhs,
        const int16_t *rhs,
        int16_t *result,
        const int32_t quan_multiplier_lhs,
        const uint8_t right_shift_lhs,
        const int32_t quan_multiplier_rhs,
        const uint8_t right_shift_rhs,
        const uint16_t col
) {
    for (uint16_t col_idx = 0; col_idx < col; ++col_idx) {
        int64_t res_lhs = quan_multiply(lhs[col_idx], quan_multiplier_lhs, right_shift_lhs);
        int64_t res_rhs = quan_multiply(rhs[col_idx], quan_multiplier_rhs, right_shift_rhs);
        int64_t res = res_lhs + res_rhs;
        res = res > int16_min ? res : int16_min;
        res = res < int16_max ? res : int16_max;
        result[col_idx] = static_cast<int16_t>(res);
    }
}

void setup() {
    Serial.begin(500000);
    Serial.setTimeout(1); 
}

void loop() {
    unsigned long timer = millis(); 

    for (int iter = 0; iter < 1000; ++iter) {
        // This is a sample code to measure the time requirement. You can modify these two variables with `read_feature` method to check its result on other input. 
        memset(tiny_light::l1_lhs_1, 0, sizeof(int16_t) * DIM_INPUT_1);
        memset(tiny_light::l1_lhs_2, 0, sizeof(int16_t) * DIM_INPUT_2);
    
        linear_relu(
                tiny_light::l1_lhs_1,
                tiny_light::l1_rhs_1,
                tiny_light::l1_bias_1,
                tiny_light::l1_sub_1,
                tiny_light::l1_quan_multiplier_w_1,
                tiny_light::l1_quan_right_shift_w_1,
                tiny_light::l1_quan_multiplier_b_1,
                tiny_light::l1_quan_right_shift_b_1,
                DIM_INPUT_1,
                DIM_LAYER_1
                );
        linear_relu(
                tiny_light::l1_lhs_2,
                tiny_light::l1_rhs_2,
                tiny_light::l1_bias_2,
                tiny_light::l1_sub_2,
                tiny_light::l1_quan_multiplier_w_2,
                tiny_light::l1_quan_right_shift_w_2,
                tiny_light::l1_quan_multiplier_b_2,
                tiny_light::l1_quan_right_shift_b_2,
                DIM_INPUT_2,
                DIM_LAYER_1
                );
        add(
                tiny_light::l1_sub_1,
                tiny_light::l1_sub_2,
                tiny_light::l1_sum,
                tiny_light::l1_quan_multiplier_sub1,
                tiny_light::l1_quan_right_shift_sub1,
                tiny_light::l1_quan_multiplier_sub2,
                tiny_light::l1_quan_right_shift_sub2,
                DIM_LAYER_1
                );
        linear_relu(
                tiny_light::l1_sum,
                tiny_light::l2_rhs,
                tiny_light::l2_bias,
                tiny_light::l2_result,
                tiny_light::l2_quan_multiplier_w,
                tiny_light::l2_quan_right_shift_w,
                tiny_light::l2_quan_multiplier_b,
                tiny_light::l2_quan_right_shift_b,
                DIM_LAYER_1,
                DIM_LAYER_2
                );
        linear(
                tiny_light::l2_result,
                tiny_light::l3_rhs,
                tiny_light::l3_bias,
                tiny_light::l3_result,
                tiny_light::l3_quan_multiplier_w,
                tiny_light::l3_quan_right_shift_w,
                tiny_light::l3_quan_multiplier_b,
                tiny_light::l3_quan_right_shift_b,
                DIM_LAYER_2,
                DIM_OUTPUT
                );
    
        uint8_t action = 0;
        for (uint8_t idx = 1; idx < DIM_OUTPUT; ++idx) {
            if (tiny_light::l3_result[idx] > tiny_light::l3_result[action]) {
                action = idx;
            }
        }
    }

    timer = millis() - timer; 
    Serial.println("time");
    Serial.println(timer); 
}
