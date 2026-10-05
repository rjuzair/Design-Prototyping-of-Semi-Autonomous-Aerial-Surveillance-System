__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void Difference(__read_only image2d_t in1, __read_only image2d_t in2, __write_only image2d_t result, float threshold)
{

	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	const float pxVal1 = read_imagef(in1, sampler, (int2)(x, y)).s0;
	const float pxVal2 = read_imagef(in2, sampler, (int2)(x, y)).s0;
	
if((pxVal1 - pxVal2) > (threshold/100)){
	write_imagef(result, (int2)(x, y), (float)((1.0)));
}
else{
	write_imagef(result, (int2)(x, y), (float)((0.0)));
}
}

