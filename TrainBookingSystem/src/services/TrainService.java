package services;

import models.Train;
import repository.TrainRepository;

import java.util.List;

public class TrainService {
    private final TrainRepository trainRepository;

    public TrainService(TrainRepository trainRepository) {
        this.trainRepository = trainRepository;
    }

    public List<Train> getAllTrains() {
        return trainRepository.findAll();
    }

    public List<Train> searchTrains(String source, String destination) {
        return trainRepository.search(source, destination);
    }

    public Train getTrainById(String trainId) {
        return trainRepository.findById(trainId)
                .orElseThrow(() -> new IllegalArgumentException("Train not found for ID: " + trainId));
    }

}
