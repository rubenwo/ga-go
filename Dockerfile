FROM golang:1.15 AS builder

WORKDIR /go/ga
COPY . .

RUN go mod download

RUN CGO_ENABLED=0 GOOS=linux go build -o ga ./cmd/ga

FROM alpine:latest
RUN apk --no-cache add ca-certificates

WORKDIR /root/
COPY --from=builder /go/ga/ga .

CMD [ "./ga" ]