/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iterator>


#include "particle_filter.h"

using namespace std;

// Helper Functions Prototypes
double calculateDistance(double x1, double y1, double x2, double y2);
void getInRangeLandmarks(Particle p, double sensor_range, Map& map_landmarks, std::vector<LandmarkObs>& in_range_landmarks);
void transformObservationsToMapCoordiantes(Particle p, std::vector<LandmarkObs>& transformed_observations);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles
	num_particles = 7;

	// set not initialized
	is_initialized = false;

	// noise generation
	default_random_engine gen;
	normal_distribution<double> N_x(0, std[0]);
	normal_distribution<double> N_y(0, std[1]);
	normal_distribution<double> N_theta(0, std[2]);

	// initialize all particles to first position
	for(int i = 0; i < num_particles; i++){
		// create new particle
		Particle p;

		// set particle attributes to first position
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1;

		// add noise to particle
		p.x += N_x(gen);
		p.y += N_y(gen);
		p.theta += N_theta(gen);

		// add particle to particle vector
		particles.push_back(p);

		// initialize weight vector with ones
		weights.push_back(1);
	}

	// set initialized
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// noise generation
	default_random_engine gen;
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; i++){
		// initialize measurements
		double meas_x = 0;
		double meas_y = 0;
		double meas_theta = 0;

		// get old theta measurement
		double theta0 = particles[i].theta;

		// calculate measurement
		if(yaw_rate == 0){
			meas_x = velocity * cos(theta0) * delta_t;
			meas_y = velocity * sin(theta0) * delta_t;
			meas_theta = 0;
		}
		else{

			meas_x = ( velocity / yaw_rate ) * ( sin(theta0 + yaw_rate * delta_t) - sin(theta0) );
			meas_y = ( velocity / yaw_rate ) * ( cos(theta0) - cos(theta0 + yaw_rate * delta_t) );
			meas_theta = yaw_rate * delta_t;
		}

		// add measurement and noise to particle position
		particles[i].x += meas_x + N_x(gen);
		particles[i].y += meas_y + N_y(gen);
		particles[i].theta += meas_theta + N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
	// find the observation with the minimum distance to predicted measurement
	for(uint i = 0; i < predicted.size(); i++){

		// set first observation as minimum distance
		int min_dist_obs_index = 0;
		double min_dist = calculateDistance(predicted[i].x, predicted[i].y, observations[0].x, observations[0].y);

		for(uint j = 1; j < observations.size(); j++){
			// calculate next observation distance
			double temp_dist = calculateDistance(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);

			if(temp_dist < min_dist){
				// found a smaller distance
				min_dist_obs_index = j;
				min_dist = temp_dist;
			}
		}

		// assign the observed measurement to this particular landmark
		predicted[i] = observations[min_dist_obs_index];
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// gaussian distribution parameters
	double std_dev_x = std_landmark[0];
	double std_dev_y = std_landmark[1];
	double var_x = std_dev_x;
	double var_y = std_dev_y;
	double c = 1/(2*M_PI*std_dev_x*std_dev_y);

	for(int i = 0; i < num_particles; i++){
		// create a vector of landmarks that are in the sensor_range of the particle
		vector<LandmarkObs> in_range_landmarks;

		// fill the in_range_landmarks with landmarks that are in range
		getInRangeLandmarks(particles[i], sensor_range, map_landmarks, in_range_landmarks);

		// initialize particle weight
		double weight = 1;

		if( in_range_landmarks.size() > 0){
			// transform the observations to the MAP's coordinate system
			std::vector<LandmarkObs> transformed_observations = observations;
			transformObservationsToMapCoordiantes(particles[i], transformed_observations);

			// associate observations with landmarks
			std::vector<LandmarkObs> associated_landmarks = transformed_observations;
			dataAssociation(associated_landmarks, in_range_landmarks);

			for(uint j = 0; j < transformed_observations.size(); j++){
				// get multi-variate gaussian distribution parameters
				double mew_x = associated_landmarks[j].x;
				double mew_y = associated_landmarks[j].y;
				double x = transformed_observations[j].x;
				double y = transformed_observations[j].y;

				// calculate distribution subterms
				double exp_x = ((x-mew_x)*(x-mew_x))/(2*var_x);
				double exp_y = ((y-mew_y)*(y-mew_y))/(2*var_y);

				// calculate new weight
				weight *= c * exp( -exp_x - exp_y);
			}

			// set new weight of particle
			particles[i].weight = weight;
		}
		else{
		  // set weight to zero when no landmarks exist
		  weight = 0;
		}
		// add new weight
		weights[i] = weight;
	}

}

void ParticleFilter::resample() {
	// create generator and discrete distribution
	default_random_engine generator;
	discrete_distribution<int> distribution( weights.begin(), weights.end()) ;

	// resample particles
	std::vector<Particle> resampled_particles;
	for(int i = 0; i < num_particles; i++){
		int index = distribution(generator);
		resampled_particles.push_back(particles[index]);
	}

	// assign resampled_particles to particles
	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

double calculateDistance(double x1, double y1, double x2, double y2){

	// calculate x positions and y positions difference
	double x_diff = x1 - x2;
	double y_diff = y1 - y2;

	// return distance
	return sqrt(x_diff*x_diff + y_diff*y_diff);
}


void getInRangeLandmarks(Particle p, double sensor_range, Map& map_landmarks, std::vector<LandmarkObs>& in_range_landmarks){

	// particle position
	double x1 = p.x;
	double y1 = p.y;

	// loop over all landmarks
	for(uint i = 0; i < map_landmarks.landmark_list.size(); i++){
		// get distance of particle to landmark
		double x2 = map_landmarks.landmark_list[i].x_f;
		double y2 = map_landmarks.landmark_list[i].y_f;
		double dist = calculateDistance(x1, y1, x2, y2);

		if( dist <= sensor_range){
			// create new LandmarkObs
			LandmarkObs obs;
			obs.id = map_landmarks.landmark_list[i].id_i;
			obs.x = x2;
			obs.y = y2;

			// add landmark to in_range_observations
			in_range_landmarks.push_back(obs);
		}
	}

}


void transformObservationsToMapCoordiantes(Particle p, std::vector<LandmarkObs>& transformed_observations){
	// set translation coordinates
	double xt = p.x;
	double yt = p.y;
	double theta = p.theta;

	// transform all observations, assume map's y-axis points downwards
	for(uint i = 0; i < transformed_observations.size(); i++){
		// get observation coordinate
		double x = transformed_observations[i].x;
		double y = transformed_observations[i].y;

		// transform
		transformed_observations[i].x = x * cos(theta) - y * sin(theta) + xt;
		transformed_observations[i].y = x * sin(theta) + y * cos(theta) + yt;
	}
}
